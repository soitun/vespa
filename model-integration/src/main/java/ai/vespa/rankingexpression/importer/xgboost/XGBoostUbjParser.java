// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.rankingexpression.importer.xgboost;

import com.devsmart.ubjson.UBArray;
import com.devsmart.ubjson.UBObject;
import com.devsmart.ubjson.UBReader;
import com.devsmart.ubjson.UBValue;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/**
 * Parser for XGBoost models in Universal Binary JSON (UBJ) format.
 *
 * @author arnej
 */
class XGBoostUbjParser {

    private final List<XGBoostTree> xgboostTrees;
    private final double baseScore;

    /**
     * Constructor stores parsed UBJ trees.
     *
     * @param filePath XGBoost UBJ input file.
     * @throws IOException Fails file reading or UBJ parsing.
     */
    XGBoostUbjParser(String filePath) throws IOException {
        this.xgboostTrees = new ArrayList<>();
        double tmpBaseScore = 0.5; // default value
        try (FileInputStream fileStream = new FileInputStream(filePath);
             UBReader reader = new UBReader(fileStream)) {
            UBValue root = reader.read();

            // Navigate to the trees array in the nested structure
            UBArray forestArray;
            if (root.isArray()) {
                // Simple array format (like JSON export)
                forestArray = root.asArray();
            } else if (root.isObject()) {
                // Nested object format: root.learner.gradient_booster.model.trees
                UBObject rootObj = root.asObject();
                UBValue learnerValue = rootObj.get("learner");
                if (learnerValue == null || !learnerValue.isObject()) {
                    throw new IOException("Expected 'learner' object in UBJ root");
                }
                UBObject learner = learnerValue.asObject();

                // Extract base_score from learner_model_param
                UBValue learnerModelParamValue = learner.get("learner_model_param");
                if (learnerModelParamValue != null && learnerModelParamValue.isObject()) {
                    UBObject learnerModelParam = learnerModelParamValue.asObject();
                    UBValue baseScoreValue = learnerModelParam.get("base_score");
                    if (baseScoreValue != null && baseScoreValue.isString()) {
                        String baseScoreStr = baseScoreValue.asString();
                        // Parse string like "[6.274165E-1]" - remove brackets and parse
                        baseScoreStr = baseScoreStr.replace("[", "").replace("]", "");
                        tmpBaseScore = Double.parseDouble(baseScoreStr);
                    }
                }
                UBValue gbValue = learner.get("gradient_booster");
                if (gbValue == null || !gbValue.isObject()) {
                    throw new IOException("Expected 'gradient_booster' object in learner");
                }
                UBObject gradientBooster = gbValue.asObject();
                UBValue modelValue = gradientBooster.get("model");
                if (modelValue == null || !modelValue.isObject()) {
                    throw new IOException("Expected 'model' object in gradient_booster");
                }
                UBObject model = modelValue.asObject();
                UBValue treesValue = model.get("trees");
                if (treesValue == null || !treesValue.isArray()) {
                    throw new IOException("Expected 'trees' array in model");
                }
                forestArray = treesValue.asArray();
            } else {
                throw new IOException("Expected UBJ array or object at root, got: " + root.getClass().getSimpleName());
            }

            // Parse each tree (UBJ format uses flat arrays, not nested objects)
            for (int i = 0; i < forestArray.size(); i++) {
                UBValue treeValue = forestArray.get(i);
                if (!treeValue.isObject()) {
                    throw new IOException("Expected UBJ object for tree, got: " + treeValue.getClass().getSimpleName());
                }
                this.xgboostTrees.add(convertUbjTree(treeValue.asObject()));
            }
        }
        this.baseScore = tmpBaseScore;
    }

    /**
     * Converts parsed UBJ trees to Vespa ranking expressions.
     *
     * @return Vespa ranking expressions.
     */
    String toRankingExpression() {
        StringBuilder ret = new StringBuilder();
        for (int i = 0; i < xgboostTrees.size(); i++) {
            ret.append(treeToRankExp(xgboostTrees.get(i)));
            if (i != xgboostTrees.size() - 1) {
                ret.append(" + \n");
            }
        }
        // Add base_score logit transformation
        ret.append(" + \n");
        ret.append("log(" + baseScore + ") - log(" + (1.0 - baseScore) + ")");
        return ret.toString();
    }

    /**
     * Recursive helper function for toRankingExpression().
     *
     * @param node XGBoost tree node to convert.
     * @return Vespa ranking expression for input node.
     */
    private String treeToRankExp(XGBoostTree node) {
        if (node.isLeaf()) {
            return Double.toString(node.getLeaf());
        } else {
            assert node.getChildren().size() == 2;
            String trueExp;
            String falseExp;
            if (node.getYes() == node.getChildren().get(0).getNodeid()) {
                trueExp = treeToRankExp(node.getChildren().get(0));
                falseExp = treeToRankExp(node.getChildren().get(1));
            } else {
                trueExp = treeToRankExp(node.getChildren().get(1));
                falseExp = treeToRankExp(node.getChildren().get(0));
            }
            // xgboost uses float only internally, so round to closest float
            float xgbSplitPoint = (float)node.getSplit_condition();
            // but Vespa expects rank profile literals in double precision:
            double vespaSplitPoint = xgbSplitPoint;
            String condition;
            if (node.getMissing() == node.getYes()) {
                // Note: this is for handling missing features, as the backend handles comparison with NaN as false.
                condition = "!(" + node.getSplit() + " >= " + vespaSplitPoint + ")";
            } else {
                condition = node.getSplit() + " < " + vespaSplitPoint;
            }
            return "if (" + condition + ", " + trueExp + ", " + falseExp + ")";
        }
    }

    /**
     * Converts a UBJ tree (flat array format) to the root XGBoostTree node (hierarchical format).
     *
     * @param treeObj UBJ object containing flat arrays representing the tree.
     * @return Root XGBoostTree node with hierarchical structure.
     */
    private static XGBoostTree convertUbjTree(UBObject treeObj) {
        // Extract flat arrays from UBJ format
        int[] leftChildren = treeObj.get("left_children").asInt32Array();
        int[] rightChildren = treeObj.get("right_children").asInt32Array();
        int[] parents = treeObj.get("parents").asInt32Array();
        float[] splitConditions = treeObj.get("split_conditions").asFloat32Array();
        int[] splitIndices = treeObj.get("split_indices").asInt32Array();
        float[] baseWeights = treeObj.get("base_weights").asFloat32Array();

        // default_left is stored as bytes/array, convert to boolean array
        byte[] defaultLeftBytes;
        UBValue defaultLeftValue = treeObj.get("default_left");
        if (defaultLeftValue.isArray()) {
            // It's a UBArray, iterate and convert
            UBArray defaultLeftArray = defaultLeftValue.asArray();
            defaultLeftBytes = new byte[defaultLeftArray.size()];
            for (int i = 0; i < defaultLeftArray.size(); i++) {
                defaultLeftBytes[i] = defaultLeftArray.get(i).asByte();
            }
        } else {
            defaultLeftBytes = defaultLeftValue.asByteArray();
        }

        // Convert from flat arrays to hierarchical tree structure
        return buildTreeFromArrays(0, leftChildren, rightChildren, parents, splitConditions,
                splitIndices, baseWeights, defaultLeftBytes);
    }

    /**
     * Recursively builds a hierarchical XGBoostTree from flat arrays.
     *
     * @param nodeId Current node index in the arrays.
     * @param leftChildren Array of left child indices.
     * @param rightChildren Array of right child indices.
     * @param parents Array of parent indices.
     * @param splitConditions Array of split threshold values.
     * @param splitIndices Array of feature indices to split on.
     * @param baseWeights Array of base weights (leaf values).
     * @param defaultLeft Array indicating if missing values go left.
     * @return XGBoostTree node.
     */
    private static XGBoostTree buildTreeFromArrays(int nodeId, int[] leftChildren, int[] rightChildren,
                                                   int[] parents, float[] splitConditions,
                                                   int[] splitIndices, float[] baseWeights,
                                                   byte[] defaultLeft) {
        XGBoostTree node = new XGBoostTree();
        setField(node, "nodeid", nodeId);

        // Calculate depth by traversing up to root
        // Note: root node has parent set to Integer.MAX_VALUE or -1
        int depth = 0;
        int currentId = nodeId;
        while (currentId >= 0 && currentId < parents.length) {
            int parentId = parents[currentId];
            if (parentId == -1 || parentId == Integer.MAX_VALUE || parentId >= parents.length) {
                break;  // Reached root
            }
            depth++;
            currentId = parentId;
        }
        setField(node, "depth", depth);

        // Check if this is a leaf node
        boolean isLeaf = leftChildren[nodeId] == -1;

        if (isLeaf) {
            // Leaf node: set the leaf value from base_weights
            // Apply float rounding to match XGBoost's internal precision
            double leafValue = baseWeights[nodeId];
            setField(node, "leaf", leafValue);
        } else {
            // Split node: set split information
            int featureIdx = splitIndices[nodeId];
            setField(node, "split", "attribute(features," + featureIdx + ")");
            // Apply float rounding to match XGBoost's internal precision (same as XGBoostParser)
            double splitValue = splitConditions[nodeId];
            setField(node, "split_condition", splitValue);

            int leftChild = leftChildren[nodeId];
            int rightChild = rightChildren[nodeId];
            boolean goLeftOnMissing = defaultLeft[nodeId] != 0;

            // In XGBoost trees:
            // - Left child is taken when feature < threshold
            // - Right child is taken when feature >= threshold
            // - default_left only controls where missing values go
            setField(node, "yes", leftChild);   // yes = condition is true = feature < threshold = go left
            setField(node, "no", rightChild);   // no = condition is false = feature >= threshold = go right
            setField(node, "missing", goLeftOnMissing ? leftChild : rightChild);

            // Recursively build children
            List<XGBoostTree> children = new ArrayList<>();
            children.add(buildTreeFromArrays(leftChild, leftChildren, rightChildren, parents,
                    splitConditions, splitIndices, baseWeights, defaultLeft));
            children.add(buildTreeFromArrays(rightChild, leftChildren, rightChildren, parents,
                    splitConditions, splitIndices, baseWeights, defaultLeft));
            setField(node, "children", children);
        }

        return node;
    }

    /**
     * Uses reflection to set a private field on an object.
     *
     * @param obj Object to modify.
     * @param fieldName Name of the field to set.
     * @param value Value to set.
     */
    private static void setField(Object obj, String fieldName, Object value) {
        try {
            Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(obj, value);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Failed to set field '" + fieldName + "' via reflection", e);
        }
    }

}

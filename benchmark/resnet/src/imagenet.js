import _ from 'lodash';
import {imagenetClasses} from '../data/imagenetClasses';
/**
 * Find top k imagenet classes
 */
export function imagenetClassesTopK(classProbabilities, k) {
    if (!k) { k = 5; }
    const probs = Array.from(classProbabilities);
    const sorted = _.reverse(_.sortBy(probs.map(function (prob, index) { return [prob, index]; }), function (probIndex) { return probIndex[0]; }));
    const topK = _.take(sorted, k).map(function (probIndex) {
        const iClass = imagenetClasses[probIndex[1]];
        return {
            id: iClass[0],
            index: parseInt(probIndex[1], 10),
            name: iClass[1].replace(/_/g, ' '),
            probability: probIndex[0]
        };
    });
    return topK;
}

import { SupportedModels } from './types';
export declare function getKeypointIndexByContour(model: SupportedModels): {
    [label: string]: number[];
};
export declare function getAdjacentPairs(model: SupportedModels): number[][];

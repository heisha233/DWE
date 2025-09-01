import numpy as np

def dataset_cfg(dataet_name):

    config = {
        'Ultrasound-Nerve':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'MEAN': [0.396269],
                'STD': [0.222191],
                'MEAN_DB2_H': [0.620777],
                'STD_DB2_H': [0.058462],
                'MEAN_DB2_L': [0.385239],
                'STD_DB2_L': [0.211054],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'GlaS':
            {
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.787803, 0.512017, 0.784938],
                'STD': [0.167663, 0.248380, 0.132252],
                'MEAN_DB2_H': [0.506244, 0.506225, 0.506242],
                'STD_DB2_H': [0.069711, 0.073610, 0.071903],
                'MEAN_DB2_L': [0.759923, 0.482532, 0.757133],
                'STD_DB2_L': [0.164817, 0.243914, 0.127698],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'EM':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'MEAN': [0.494758],
                'STD': [0.172635],
                'MEAN_DB2_H': [0.500624],
                'STD_DB2_H': [0.100202],
                'MEAN_DB2_L': [0.495944],
                'STD_DB2_L': [0.164370],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'ROSSA':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'MEAN': [0.355049],
                'STD': [0.274810],
                'MEAN_DB2_H': [0.490757],
                'STD_DB2_H': [0.118064],
                'MEAN_DB2_L': [0.362650],
                'STD_DB2_L': [0.220478],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            }
    }

    return config[dataet_name]

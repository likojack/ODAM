import json
import numpy as np
import os
import pickle


SCAN2CAD_DIR = "/home/kejie/ext_disk/datasets/Scan2CAD"

CARE_CLASSES = {
    "03211117": "display",
    "04379243": "table",
    "02808440": "bathtub",
    "02747177": "trashbin",
    "04256520": "sofa",
    "03001627": "chair",
    "02933112": "cabinet",
    "02871439": "bookshelf",
}


def main():
    sizes_per_class = {k: [] for k in CARE_CLASSES}
    with open(os.path.join(SCAN2CAD_DIR, "full_annotations.json"), "r") as f:
        scans = json.load(f)
    for scan in scans:
        for model in scan['aligned_models']:
            if not model['catid_cad'] in CARE_CLASSES:
                continue
            scales = model['trs']['scale']
            scales = model['bbox'] * np.asarray(scales) * 2
            scales = scales[[2, 0, 1]]
            sizes_per_class[model['catid_cad']].append(scales)

    out_dict = {}
    for k in CARE_CLASSES:
        sizes = np.asarray(sizes_per_class[k])
        covariance = np.linalg.inv(np.cov(sizes, rowvar=False))
        mean = np.mean(sizes, axis=0)
        out_dict[k] = covariance
        print(CARE_CLASSES[k])
        print("mean: ")
        print(mean)
        print("inverse covariance: ")
        print(covariance)
        print("---------")
    with open("./src/super_quadric/scale_prior", "wb") as f:
        pickle.dump(out_dict, f)


if __name__ == "__main__":
    main()
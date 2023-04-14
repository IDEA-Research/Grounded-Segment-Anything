## Removing Chumpy objects

In a Python 2 virtual environment with [Chumpy](https://github.com/mattloper/chumpy) installed run the following to remove any Chumpy objects from the model data:

```bash
python tools/clean_ch.py --input-models path-to-models/*.pkl --output-folder output-folder
```

## Merging SMPL-H and MANO parameters

In order to use the given PyTorch SMPL-H module we first need to merge the SMPL-H and MANO parameters in a single file. After agreeing to the license and downloading the models, run the following command:

```bash
python tools/merge_smplh_mano.py --smplh-fn SMPLH_FOLDER/SMPLH_GENDER.pkl \
 --mano-left-fn MANO_FOLDER/MANO_LEFT.pkl \
 --mano-right-fn MANO_FOLDER/MANO_RIGHT.pkl \
 --output-folder OUTPUT_FOLDER
```

where SMPLH_FOLDER is the folder with the SMPL-H files and MANO_FOLDER the one for the MANO files.

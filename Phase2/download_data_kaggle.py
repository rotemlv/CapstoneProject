
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
# install this one!
from pyuac import main_requires_admin


@main_requires_admin
def main():
    CHUNK_SIZE = 40960
    DATA_SOURCE_MAPPING =  'uw-madison-gi-tract-image-segmentation:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F27923%2F3495119%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240728%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240728T155120Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3Db4b54e463e04b14a5e5d02d069b3472081053be289768fe81456a8c4f541f6f28cf4aa29afdc8fcd4e611ce52b46a64467213c05ae18270016b275cd6b6a6890eafd372496e819df86fd43f712ffac6231633c5edab53d9f79d86adeffd0672bf92fe2e2dd948e6861e03d8d637a9f16464b42d866aac6f9d90beab8d9e8271e2dbf765c13cc14045ac2c5b14d7e34651ad310c3379a19f6309ccfbbf09603d0d87c7c8d427ef7a4bafc9e13223ceed97b655fbad86b58cebe250c1169692e2dd626b28493311af621a21b777cca5051b2ddcf0b66e64a15c48f9e07375c402bab588eb2c23748507758b6268ca9c787c10f4ed2cbcfcad7412bf39a1aebecd7,pytorch-segmentation-models-lib:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F1701116%2F2786089%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240728%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240728T155120Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D55d2cc42da223679a44bb083c661b896468319ed5a7a2a9e3dc7d1ce68be47a12ab765821588b947039751df939d24e8c79957bc605447358aced4563b5ed9ec8635e5fcd98c7dc743c2bef48088d0ebb4df2f0c44a3c47be53dc820faa93584dba4dab7e1768a024fdba1fc2d4dcdbaf43f556f942dbaa5f821b1d12d10577907fc5a7a448fea536212e5bc526d296ebcb76c06d4af2e529f057220f81c7c74a22fe8cc400d80c19ca33eaef219dcb7e427cbe920155b17915f236a13484e138cfc8c8869c43c45b28539af692afee9930eddeb4938f7e7343b7900a6b1e468799c46fc5be296089945a5ddf95d7e6f4236498427069afcfaaafa59a0885ac7,uwmgi-mask-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F2089184%2F3521209%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240728%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240728T155120Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D3f073c5645f1f5efd872ae9c2c23d38875a3447a0d9acd1cf0ca095ed0856808ca0b17aea3cb05bf67b46296fe556cb69ab3e619cdddd1b00c5926fd04f41acc61a683038ec5bf0488d19afd257ee08a6e32e56758374a104eccd0dec847e0dd8460e22ddc41453823c446f99362a8e975d1568980c2fb44699cb3a48b73e07e717f1e5f5c1c1d8e6a150b744dc4fe96d339ff2488a044fab65716c14932a33bb756562082d55f500d4c71c9158f79ead158ad869135d6ea956a851c210ed93bbb6f1a768321f46b0cc723d32b5fadc68202a005b16c544c3b166769f12ff731d5cd2938a56dd4e023dacb37d6883e539580638cf3301bc7e79d219cb54999f0,uwmgi-25d-stride2-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F2142905%2F3566043%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240728%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240728T155120Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D9b752b5c5a42175b0a730c54cf4b47fe5ed231cb3fe4fe1d099e572b196ff8e6d2b138a708b82a0bd299f9073ad1239975e947600468e355b41cecf000de5f06710d64a964794c68847dadccbb7c6c106f36c7e44afa4d60fdcad2129660fcc9edb73895cf7c2020135e9f620a00e7fc4204a625349eddb1923f0a2c73077b01dac21938ae6771f69888907efc8a212782c478cb0731d8c64794017e23a278f3b3e5aa3a3cf059a4c2ab4567febec9679d365f899d8db68afbde62632dcbef82cfbb82e5220951fcd9eb5f65176597d7dc0817f9060d617cabe95124361134f4e8659aaeda438cba36ed3b5b96c5fe8fe67edc3d1c22f6c6d24d1a3a8aa48cce'

    KAGGLE_INPUT_PATH='/kaggle/input'
    KAGGLE_WORKING_PATH='/kaggle/working'
    KAGGLE_SYMLINK='kaggle'

    shutil.rmtree('/kaggle/input', ignore_errors=True)
    os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
    os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

    try:
      os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
    except FileExistsError:
      pass
    try:
      os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
    except FileExistsError:
      pass

    for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
        directory, download_url_encoded = data_source_mapping.split(':')
        download_url = unquote(download_url_encoded)
        filename = urlparse(download_url).path
        destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
        try:
            with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
                total_length = fileres.headers['content-length']
                print(f'Downloading {directory}, {total_length} bytes compressed')
                dl = 0
                data = fileres.read(CHUNK_SIZE)
                while len(data) > 0:
                    dl += len(data)
                    tfile.write(data)
                    done = int(50 * dl / int(total_length))
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                    sys.stdout.flush()
                    data = fileres.read(CHUNK_SIZE)
                if filename.endswith('.zip'):
                  with ZipFile(tfile) as zfile:
                    zfile.extractall(destination_path)
                else:
                  with tarfile.open(tfile.name) as tarfile:
                    tarfile.extractall(destination_path)
                print(f'\nDownloaded and uncompressed: {directory}')
        except HTTPError as e:
            print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
            continue
        except OSError as e:
            print(f'Failed to load {download_url} to path {destination_path}')
            continue

    print('Data source import complete.')


if __name__ == '__main__':
    main()

# riscureTrsToCsv

trsConverter.py

To run:

1. Scenario for Fixed Key (Test Set)
python trsConverter.py filename.trs AES -k deadbeef01234567cafebabe89abcdef

2. Scenario for Random Key (Train Set)
python trsConverter.py filename.trs AES

Output will be traces.trs and will contain:

plaintext traceSamples* ciphertext key leakages


template_deepnet_aes.py

To run:

python template_deepnet_aes.py

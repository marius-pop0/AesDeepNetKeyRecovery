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

Will run only if GPU is available (use optirun, cuda, cudnn...)

To run:

usage: template_deepnet_aes.py [-h] [-e E] [-ms MS] [-ts TS] [-vs VS] [-r R]
                               [-rl RL]
                               {mlp,cnn} a
                               {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

Deepnet AES Side Channel Analysis

positional arguments:

  {mlp,cnn}             Network Type
  
  a                     Network Architecture - [[convBlock1],[ConvBlock2],[ConvBlock...],[DenseLayers]]
                        
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
                        Key Byte To Attack

optional arguments:

  -h, --help            show this help message and exit

  -e E                  Number of Epochs (int)

  -ms MS                Mutual Information Samples (int)

  -ts TS                Train Set Size (int)

  -vs VS                Validation Set Size (int)

  -r R                  Repetitions (int)

  -rl RL                Random Labels (bool)

eg. optirun python template_deepnet_aes.py cnn [[16,16],[16],[128,64,32,16]] 0 -e 200 -ms 75 -r 5
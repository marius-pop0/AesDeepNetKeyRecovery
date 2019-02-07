import pandas as pd
import numpy as np
import trsfile
import binascii
import pyDesLoc
import aes

# Modify to be taken as args
DES: int = 0
AES: int = 1
# Keys for each algorithm
key_a = 'deadbeef01234567cafebabe89abcdef'
key_d = bytes.fromhex('deacbeeecafebabe')
filename = 'AES + StaticAlign + LowPass + POI.trs'
# Alg class instantiation
k = pyDesLoc.des(key_d, pyDesLoc.ECB)
a = aes.AES(mode='ecb', input_type='hex')

ALG = AES
#

with trsfile.open(filename, 'r') as traces:
    # Show all headers
    for header, value in traces.get_headers().items():
        print(header, '=', value)

    df_traces = pd.DataFrame()
    df_data = pd.DataFrame()

    # Iterate over the traces
    for i, trace in enumerate(traces):
        # Print Trace and Info
        # print(pd.DataFrame([trace.samples]))
        # print('Trace {0:d} contains {1:d} samples'.format(i, len(trace)))
        # print('  - minimum value in trace: {0:f}'.format(min(trace)))
        # print('  - maximum value in trace: {0:f}'.format(max(trace)))
        df_traces = df_traces.append(pd.DataFrame([trace.samples]), ignore_index=True)

        # Print Data
        # print(binascii.hexlify(trace.data).decode('utf8'))
        # print("Plaintext " + binascii.hexlify(trace.data).decode('utf8')[0:16])
        # print("Chipertext " + binascii.hexlify(trace.data).decode('utf8')[16:32])
        data = binascii.hexlify(trace.data).decode('utf8')

        if ALG == DES:
            # Compute DES intermediate values
            d, sbox_in, sbox_out, r_in, r_out = k.encrypt(bytes.fromhex(data[0:16]))

            # Pad Sbox_Out with 0 bits at front and back
            sbox_count = 8
            front = 0
            back = 5
            while sbox_count > 0:
                list.insert(sbox_out[0], front, 0)
                list.insert(sbox_out[0], back, 0)
                front += 6
                back += 6
                sbox_count -= 1

            # Round 1 HW -> Sbox_Out
            HW_r1_sbox = 0
            for i in sbox_out[0]:
                if i == 1:
                    HW_r1_sbox += 1

            # Round 1 HD -> Sbox_In XOR Sbox_Out
            HD_r1_sbox = 0
            sbox_xor_r1 = np.bitwise_xor(sbox_in[0], sbox_out[0])
            for i in sbox_xor_r1:
                if i == 1:
                    HD_r1_sbox += 1

            # Round 1 HW -> Round_Out
            HW_r1_rOut = 0
            for i in r_out[0]:
                if i == 1:
                    HW_r1_rOut += 1

            # Round 1 HD -> Round_In XOR Round_Out
            HD_r1_round = 0
            r_xor_r1 = np.bitwise_xor(r_in[0], r_out[0])
            for i in r_xor_r1:
                if i == 1:
                    HD_r1_round += 1

            # print("SboxHW:{}, SboxHD:{}, RoundHW:{}, RoundHD:{}".format(HW_r1_sbox, HD_r1_sbox, HW_r1_rOut,
            # HD_r1_round))
            # save the data to a data frame (Plaintext, HW-Round, HD-Round, HW-Sbox, HD-Sbox)
            df_data = df_data.append(
                pd.DataFrame([[data[0:16], data[16:32], HW_r1_sbox, HD_r1_sbox, HW_r1_rOut, HD_r1_round]]),
                ignore_index=True)

        else:
            # Compute AES intermediate values
            d, sbox_in, sbox_out = a.encryption(data[0:32], key_a)

            # Round 1 HW -> Sbox_Out
            HW_r1_sbox = 0
            for i in sbox_out[0]:
                HW_r1_sbox += bin(int(i, 16))[2:].zfill(8).count("1")

            # Round 1 HD -> Sbox_In XOR Sbox_Out
            HD_r1_sbox = 0
            sbox_xor_r1 = [None]*16
            # Bitwise Xor between intermediate values
            for i in range(0, 16):
                sbox_xor_r1[i] = int(sbox_in[0][i], 16) ^ int(sbox_out[0][i], 16)
                HD_r1_sbox += bin(sbox_xor_r1[i])[2:].zfill(8).count("1")

        # print("SboxHW:{}, SboxHD:{}, RoundHW:{}, RoundHD:{}".format(HW_r1_sbox, HD_r1_sbox))
        # save the data to a data frame (Plaintext, Ciphertext, HW-Sbox, HD-Sbox)
        df_data = df_data.append(pd.DataFrame([[data[0:32], data[32:64], HW_r1_sbox, HD_r1_sbox]]), ignore_index=True)

    # insert plaintext at the from of the dataframe
    df_traces.insert(loc=0, column='pt', value=df_data[0])
    df_traces['ct'] = df_data[1]
    df_traces['round1HW_SboxOut'] = df_data[2]
    df_traces['round1HD_SboxOut'] = df_data[3]
    if ALG == DES:
        df_traces['round1HW_RoundOut'] = df_data[4]
        df_traces['round1HD_RoundOut'] = df_data[5]

    # Sort by column name
    # df_traces = df_traces.sort_values('round1_SboxOut')

    print(df_traces.head())

    # Write to CSV file --> Takes a while to do for large traces
    df_traces.to_csv("trace.csv", index=False)

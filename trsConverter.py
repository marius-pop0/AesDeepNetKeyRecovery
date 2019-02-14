import pandas as pd
import numpy as np
import trsfile
import binascii
import des
import aes

# Modify to be taken as args
DES: int = 0
AES: int = 1
# Keys for each algorithm
key_a = 'deadbeef01234567cafebabe89abcdef'
key_d = bytes.fromhex('deacbeeecafebabe')
filename = 'HWDES+Harmonic+Resample+StaticAlign+PoiSelection.trs'
# Alg class instantiation
k = des.des(key_d, des.ECB)
a = aes.AES(mode='ecb', input_type='hex')

ALG = DES

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
            HW_r1_sbox_b = []
            HW_r1_sbox = sum(sbox_out[0])
            for b in range(0, len(sbox_out[0]), 8):
                HW_r1_sbox_b.append(sum(sbox_out[0][b:b+8]))

            # Round 1 HD -> Sbox_In XOR Sbox_Out
            HD_r1_sbox = 0
            HD_r1_sbox_b = []
            sbox_xor_r1 = np.bitwise_xor(sbox_in[0], sbox_out[0])
            HD_r1_sbox = sum(sbox_xor_r1)
            for b in range(0, len(sbox_xor_r1), 8):
                HD_r1_sbox_b.append(sum(sbox_xor_r1[b:b + 8]))


            # Round 1 HW -> Round_Out
            HW_r1_rOut = 0
            HW_r1_rOut_b = []
            HW_r1_rOut = sum(r_out[0])
            for b in range(0, len(r_out[0]), 8):
                HW_r1_rOut_b.append(sum(r_out[0][b:b+8]))

            # Round 1 HD -> Round_In XOR Round_Out
            HD_r1_round = 0
            HD_r1_round_b = []
            r_xor_r1 = np.bitwise_xor(r_in[0], r_out[0])
            HD_r1_round = sum(r_xor_r1)
            for b in range(0, len(r_xor_r1), 8):
                HD_r1_round_b.append(sum(r_xor_r1[b:b+8]))

            # print("SboxHW:{}, SboxHD:{}, RoundHW:{}, RoundHD:{}".format(HW_r1_sbox, HD_r1_sbox, HW_r1_rOut,
            # HD_r1_round))
            # save the data to a data frame (Plaintext, HW-Round, HD-Round, HW-Sbox, HD-Sbox)
            # df_data = df_data.append(
            #     pd.DataFrame([[data[0:16], data[16:32], HW_r1_sbox, HD_r1_sbox, HW_r1_rOut, HD_r1_round]]),
            #     ignore_index=True)
            df_data = df_data.append(
                pd.DataFrame([[data[0:16], data[16:32], HD_r1_round_b[0], HD_r1_round_b[1], HD_r1_round_b[2],
                               HD_r1_round_b[3], HD_r1_round_b[4], HD_r1_round_b[5], HD_r1_round_b[6], HD_r1_round_b[7]]
                              ]), ignore_index=True)
        else:
            # Compute AES intermediate values
            d, sbox_in, sbox_out = a.encryption(data[0:32], key_a)

            # Round 1 HW -> Sbox_Out
            HW_r1_sbox = 0
            HW_r1_sbox_b = []
            for i in sbox_out[0]:
                hw = bin(int(i, 16))[2:].zfill(8).count("1")
                HW_r1_sbox += hw
                HW_r1_sbox_b.append(hw)

            # Round 1 HD -> Sbox_In XOR Sbox_Out
            HD_r1_sbox = 0
            HD_r1_sbox_b = []
            sbox_xor_r1 = [None]*16
            # Bitwise Xor between intermediate values
            for i in range(0, 16):
                sbox_xor_r1[i] = int(sbox_in[0][i], 16) ^ int(sbox_out[0][i], 16)
                hd = bin(sbox_xor_r1[i])[2:].zfill(8).count("1")
                HD_r1_sbox += hd
                HD_r1_sbox_b.append(hd)

            # print("SboxHW:{}, SboxHD:{}, RoundHW:{}, RoundHD:{}".format(HW_r1_sbox, HD_r1_sbox))
            # save the data to a data frame (Plaintext, Ciphertext, HW-Sbox, HD-Sbox)
            df_data = df_data.append(pd.DataFrame([[data[0:32], data[32:64], HW_r1_sbox, HD_r1_sbox]]), ignore_index=True)

    # insert plaintext at the from of the dataframe
    df_traces.insert(loc=0, column='pt', value=df_data[0])
    df_traces['ct'] = df_data[1]
    if ALG == AES:
        df_traces['round1HW_SboxOut'] = df_data[2]
        df_traces['round1HD_SboxOut'] = df_data[3]
    if ALG == DES:
        df_traces['round1HD_RoundOut_b0'] = df_data[2]
        df_traces['round1HD_RoundOut_b1'] = df_data[3]
        df_traces['round1HD_RoundOut_b2'] = df_data[4]
        df_traces['round1HD_RoundOut_b3'] = df_data[5]
        df_traces['round1HD_RoundOut_b4'] = df_data[6]
        df_traces['round1HD_RoundOut_b5'] = df_data[7]
        df_traces['round1HD_RoundOut_b6'] = df_data[8]
        df_traces['round1HD_RoundOut_b7'] = df_data[9]


    # Sort by column name
    # df_traces = df_traces.sort_values('round1_SboxOut')

    print(df_traces.head())

    # Write to CSV file --> Takes a while to do for large traces
    df_traces.to_csv("trace.csv", index=False)

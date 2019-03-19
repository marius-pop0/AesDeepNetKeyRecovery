import pandas as pd
import numpy as np
import trsfile
import binascii
import des
import aes
import argparse

parser = argparse.ArgumentParser(description='Convert trs files to CSV.')
parser.add_argument('f', help="Input file .trs", type=str, nargs=1)
parser.add_argument('a', help="Algorithm Used. AES or DES", type=str, nargs=1, choices=['AES', 'DES'], )
parser.add_argument('-k', nargs=1, type=str, help=' Uniform Encryption Key supplied (key not in dataset)')

args = parser.parse_args()
DES: int = 0
AES: int = 1

if args.a[0] == "DES":
    ALG = DES
elif args.a[0] == "AES":
    ALG = AES
    a = aes.AES(mode='ecb', input_type='hex')
else:
    ALG = -1
    exit("Invalid Algorithm")


filename = args.f[0]

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
        # print("Key " + binascii.hexlify(trace.data).decode('utf8')[32:48])

        data = binascii.hexlify(trace.data).decode('utf8')

        if ALG == DES:
            if args.k is None:
                key = bytes.fromhex(data[32:])
                k = des.des(key, des.ECB)
            else:
                if len(args.k[0]) == 16:
                    key = bytes.fromhex(args.k[0])
                    k = des.des(key, des.ECB)
                else:
                    exit("Invalid Key Length for DES")
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
            HW_r1_sbox_b = []
            HW_r1_sbox = sum(sbox_out[0])
            for b in range(0, len(sbox_out[0]), 8):
                HW_r1_sbox_b.append(sum(sbox_out[0][b:b + 8]))

            # Round 1 HD -> Sbox_In XOR Sbox_Out
            HD_r1_sbox_b = []
            sbox_xor_r1 = np.bitwise_xor(sbox_in[0], sbox_out[0])
            HD_r1_sbox = sum(sbox_xor_r1)
            for b in range(0, len(sbox_xor_r1), 8):
                HD_r1_sbox_b.append(sum(sbox_xor_r1[b:b + 8]))

            # Round 1 HW -> Round_Out
            HW_r1_rOut_b = []
            HW_r1_rOut = sum(r_out[0])
            for b in range(0, len(r_out[0]), 8):
                HW_r1_rOut_b.append(sum(r_out[0][b:b + 8]))

            # Round 1 HD -> Round_In XOR Round_Out
            HD_r1_round_b = []
            r_xor_r1 = np.bitwise_xor(r_in[0], r_out[0])
            HD_r1_round = sum(r_xor_r1)
            for b in range(0, len(r_xor_r1), 8):
                HD_r1_round_b.append(sum(r_xor_r1[b:b + 8]))

            # check is key is in trace or supplied
            df_all = pd.DataFrame([[data[0:16], data[16:32]]])
            if args.k is None:
                df_all = pd.concat([df_all, pd.DataFrame([[data[32:48]]])], axis=1)
            for i in range(8):
                df_all = pd.concat([df_all, pd.DataFrame([[HD_r1_round_b[i]]])], axis=1)
            df_data = df_data.append(df_all)

        elif ALG == AES:
            if args.k is None:
                key = data[64:96]
            else:
                if len(args.k[0]) == 32:
                    key = args.k[0]
                else:
                    exit("Invalid Key Length for AES")

            # Compute AES intermediate values
            d, sbox_in, sbox_out = a.encryption(data[0:32], key)

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
            sbox_xor_r1 = [None] * 16
            # Bitwise Xor between intermediate values
            for i in range(0, 16):
                sbox_xor_r1[i] = int(sbox_in[0][i], 16) ^ int(sbox_out[0][i], 16)
                hd = bin(sbox_xor_r1[i])[2:].zfill(8).count("1")
                HD_r1_sbox += hd
                HD_r1_sbox_b.append(hd)

            # print("SboxHW:{}, SboxHD:{}, RoundHW:{}, RoundHD:{}".format(HW_r1_sbox, HD_r1_sbox))
            # save the data to a data frame (Plaintext, Ciphertext, HW-Sbox, HD-Sbox)
            # check if key is in trace or supplied
            df_all = pd.DataFrame([[data[0:32], data[32:64]]])
            if args.k is None:
                df_all = pd.concat([df_all,pd.DataFrame([[data[64:96]]])], axis=1)
            for i in range(16):
                df_all = pd.concat([df_all, pd.DataFrame([[HW_r1_sbox_b[i]]])], axis=1)

            df_data = df_data.append(df_all)

    df_data = df_data.reset_index(drop=True)
    df_data.columns = range(df_data.shape[1])
    # insert plaintext at the from of the dataframe
    df_traces.insert(loc=0, column='pt', value=df_data[0])
    df_traces['ct'] = df_data[1]

    base_max_idx_aes = 18
    base_max_idx_des = 10

    if args.k is None:
        df_traces['key'] = df_data[2]
        idx = 3
        base_max_idx_aes += 1
        base_max_idx_des += 1
    else:
        df_traces['key'] = args.k[0]
        idx = 2

    b = 0
    if ALG == AES:
        while idx < base_max_idx_aes:
            col = 'round1HW_SboxOut_b{}'.format(b)
            df_traces[col] = df_data[idx]
            idx += 1
            b += 1

    if ALG == DES:
        while idx < base_max_idx_des:
            col = 'round1HD_RoundOut_b{}'.format(b)
            df_traces[col] = df_data[idx]
            idx += 1
            b += 1

    # Sort by column name
    # df_traces = df_traces.sort_values('round1_SboxOut')

    # print(df_traces.head())

    # Write to CSV file --> Takes a while to do for large traces
    df_traces.to_csv("trace.csv", index=False)

    print("Conversion and Leakage Calculations Complete!")

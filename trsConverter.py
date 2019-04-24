import pandas as pd
import numpy as np
import trsfile
import binascii
import des
import argparse

parser = argparse.ArgumentParser(description='Convert trs files to CSV.')
parser.add_argument('f', help="Input file .trs", type=str, nargs=1)
parser.add_argument('a', help="Algorithm Used. AES or DES", type=str, nargs=1, choices=['AES', 'DES'], )
parser.add_argument('-k', nargs=1, type=str, help=' Uniform Encryption Key supplied (key not in dataset)')

args = parser.parse_args()
DES: int = 0
AES: int = 1

AES_Sbox = np.array(
    [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76, 0xCA, 0x82, 0xC9,
     0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F,
     0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15, 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07,
     0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3,
     0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58,
     0xCF, 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3,
     0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC, 0x5F,
     0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73, 0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
     0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC,
     0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A,
     0xAE, 0x08, 0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 0x70,
     0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1, 0xF8, 0x98, 0x11,
     0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF, 0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42,
     0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16])

hw = np.array([bin(x).count("1") for x in range(256)])


if args.a[0] == "DES":
    ALG = DES
elif args.a[0] == "AES":
    ALG = AES
else:
    ALG = -1
    exit("Invalid Algorithm")


filename = args.f[0]

with trsfile.open(filename, 'r') as traces:
    # Show all headers
    for header, value in traces.get_headers().items():
        print(header, '=', value)

    # TODO: Should change to, to allocate empty (NaN) df since we know the sizes from the header
    # df_ = pd.DataFrame(index=index, columns=columns)
    df_traces = pd.DataFrame()
    df_data = pd.DataFrame()

    # Iterate over the traces
    for i, trace in enumerate(traces):
        if i % 1000 == 0:
            # Report Every 1000 Lines Processed
            print("Processing Line: {}".format(i))
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
            # hemw = hw[AES_Sbox[bytes.fromhex(v)[kByte] ^ kh]]
            HD_r1_sbox = np.zeros(shape=16, dtype=np.int)
            HW_r1_sbox = np.zeros(shape=16, dtype=np.int)

            for i, v in enumerate(bytes.fromhex(data[0:32])):
                sbox_in = v ^ bytes.fromhex(key)[i]
                sbox_out = AES_Sbox[sbox_in]
                HD_r1_sbox[i] = int(hw[sbox_in ^ sbox_out])
                HW_r1_sbox[i] = int(hw[sbox_out])

            # # Compute AES intermediate values
            # d, sbox_in, sbox_out = a.encryption(data[0:32], key)
            #
            # # Round 1 HW -> Sbox_Out
            # HW_r1_sbox = 0
            # HW_r1_sbox_b = []
            # for i in sbox_out[0]:
            #     hw = bin(int(i, 16))[2:].zfill(8).count("1")
            #     HW_r1_sbox += hw
            #     HW_r1_sbox_b.append(hw)
            #
            # # Round 1 HD -> Sbox_In XOR Sbox_Out
            # HD_r1_sbox = 0
            # HD_r1_sbox_b = []
            # sbox_xor_r1 = [None] * 16
            # # Bitwise Xor between intermediate values
            # for i in range(0, 16):
            #     sbox_xor_r1[i] = int(sbox_in[0][i], 16) ^ int(sbox_out[0][i], 16)
            #     hd = bin(sbox_xor_r1[i])[2:].zfill(8).count("1")
            #     HD_r1_sbox += hd
            #     HD_r1_sbox_b.append(hd)
            #
            # # print("SboxHW:{}, SboxHD:{}, RoundHW:{}, RoundHD:{}".format(HW_r1_sbox, HD_r1_sbox))
            # # save the data to a data frame (Plaintext, Ciphertext, HW-Sbox, HD-Sbox)
            # # check if key is in trace or supplied
            df_all = pd.DataFrame([[data[0:32], data[32:64]]])
            if args.k is None:
                df_all = pd.concat([df_all,pd.DataFrame([[data[64:96]]])], axis=1)
            for i in range(16):
                df_all = pd.concat([df_all, pd.DataFrame([[HW_r1_sbox[i]]])], axis=1)

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

import pandas as pd
import numpy as np
import trsfile
import binascii
import pyDesLoc

key = bytes.fromhex('deacbeeecafebabe')
k = pyDesLoc.des(key, pyDesLoc.ECB)

with trsfile.open('HWDES+Harmonic+Resample+StaticAlign+PoiSelection.trs', 'r') as traces:
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

        # print("SboxHW:{}, SboxHD:{}, RoundHW:{}, RoundHD:{}".format(HW_r1_sbox, HD_r1_sbox, HW_r1_rOut, HD_r1_round))
        df_data = df_data.append(pd.DataFrame([[data[0:16], data[16:32], HW_r1_sbox, HD_r1_sbox, HW_r1_rOut, HD_r1_round]]), ignore_index=True)

    df_traces.insert(loc=0, column='pt', value=df_data[0])
    df_traces['ct'] = df_data[1]
    df_traces['round1HW_SboxOut'] = df_data[2]
    df_traces['round1HD_SboxOut'] = df_data[3]
    df_traces['round1HW_RoundOut'] = df_data[4]
    df_traces['round1HD_RoundOut'] = df_data[5]

    # Sort by column name
    # df_traces = df_traces.sort_values('round1_SboxOut')

    print(df_traces.head())

    # Write to CSV file --> Takes a while to do for large traces
    df_traces.to_csv("trace.csv", index=False)

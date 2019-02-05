import pandas as pd
import trsfile
import binascii
import pyDes

key = bytes.fromhex('deacbeeecafebabe')

tmpdata = bytes.fromhex('2313fd389fcdab92')
k = pyDes.des(key,pyDes.ECB)
d,sbox_out = k.encrypt(tmpdata)
print(binascii.hexlify(d).decode('utf-8'))
print(sbox_out)
HW_r1 = 0
#Round 1 HW
for i in sbox_out[0]:
    if i == 1:
        HW_r1 += 1
print(HW_r1)

with trsfile.open('HWDES+Harmonic+Resample+StaticAlign+PoiSelection.trs', 'r') as traces:
    # Show all headers
    for header, value in traces.get_headers().items():
        print(header, '=', value)

    df_traces = pd.DataFrame()
    df_data = pd.DataFrame()
    # Iterate over the traces
    for i, trace in enumerate(traces[0:5]):
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
        df_data = df_data.append(pd.DataFrame([[data[0:16], data[16:32]]]), ignore_index=True)

    df_traces.insert(loc=0, column='pt', value=df_data[0])
    df_traces['ct'] = df_data[1]

    print(df_traces.head())
    # Write to CSV file --> Takes a while to do for large traces
    df_traces.to_csv("trace.csv", index=False)

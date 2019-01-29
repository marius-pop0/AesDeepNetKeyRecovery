import pandas as pd
import trsfile

with trsfile.open('HWDES50000Traces.trs', 'r') as traces:
    # Show all headers
    for header, value in traces.get_headers().items():
        print(header, '=', value)

    df = pd.DataFrame()

    # Iterate over the traces
    for i, trace in enumerate(traces):
        df = df.append(pd.DataFrame([trace.samples]), ignore_index=True)
        # Print Trace and Info

        # print(pd.DataFrame([trace.samples]))
        # print('Trace {0:d} contains {1:d} samples'.format(i, len(trace)))
        # print('  - minimum value in trace: {0:f}'.format(min(trace)))
        # print('  - maximum value in trace: {0:f}'.format(max(trace)))

    df.to_csv("foo.csv", index=False)

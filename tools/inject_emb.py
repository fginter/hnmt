import argparse
import pickle
import lwvlib
import sys
import numpy as np

def replace_embeddings(host,donor,out):
    """
    Replaces donor embeddings in host, writes to out.

    host: hnmt/bnas Embeddings submodel file name in
    donor: word2vec bin file name
    out: hnmt/bnas Embeddings submodel file name out
    """
    with open(host,"rb") as hostf:
        host_config=pickle.load(hostf)
        host_data=pickle.load(hostf)
        host_matrix=host_data[("w",)]
        print("...{} loaded".format(host),file=sys.stderr,flush=True)
    donor_model=lwvlib.load(donor)
    donor_matrix=donor_model.vectors.astype(host_matrix.dtype)
    assert host_matrix.shape[1]==donor_matrix.shape[1], (host_matrix.shape[1],donor_matrix.shape[1])
    print("...{} loaded".format(donor),file=sys.stderr,flush=True)
    print("Sum before exchange",np.sum(host_data[("w",)]),file=sys.stderr, flush=True)
    #Replace
    for w,host_dim in host_config["src_encoder"].index.items():
        donor_dim=donor_model.get(w)
        if donor_dim is None:
            print("Cannot map {}".format(w),file=sys.stderr,flush=True)
            continue
        host_matrix[host_dim]=donor_matrix[donor_dim]
    #Done replacing now
    with open(out,"wb") as outf:
        pickle.dump(host_config,outf)
        pickle.dump(host_data,outf)
    print("Sum after exchange",np.sum(host_data[("w",)]),file=sys.stderr, flush=True)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Inject donor embeddings into host submodel file, writing out a new submodel file')
    parser.add_argument('--host-submodel', help='A host _embeddings file produced by split-model')
    parser.add_argument('--donor-bin', help='A donor .bin (Mikolov) file with vectors of the correct dimensionality')
    parser.add_argument('--out-submodel', help='Where to write the resulting _embeddings file')
    
    args = parser.parse_args()
    replace_embeddings(args.host_submodel,args.donor_bin,args.out_submodel)

    

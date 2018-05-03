#!/usr/local/bin/python3
'''
Convert a pkl file into json file
Credits: https://gist.github.com/Samurais/567ebca0f59c612eb977065008aad867
'''
import sys
import cPickle as pickle
import json

def convert_dict_to_json(file_path, out_file_dir, feat_type):
    with open(file_path, 'rb') as fpkl, open(out_file_dir+'/'+feat_type+'_feat.json', 'w') as fjson:
    	data = pickle.load(fpkl)
        ks = data.keys()
        l = {}
        for k in ks:
            v = data[k]
            v_str = {feat_type+'_feat': ' '.join(str(x) for x in v).encode('utf_8')}
            l[k.encode('utf_8')] = v_str
        l_all = {'utts' : l}
    	json.dump(l_all, fjson, ensure_ascii=False, indent=4)

def main():
    file_path = sys.argv[1]
    out_file_dir = sys.argv[2]
    feat_type = sys.argv[3]
    convert_dict_to_json(file_path, out_file_dir, feat_type)

if __name__ == '__main__':
    main()

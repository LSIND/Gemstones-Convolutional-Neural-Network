#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

def main():
    userpath = sys.argv[1]
    try:
        with os.scandir(userpath) as rootfolder:
            for entry in rootfolder:
                if(entry.is_dir()):
                    f_path= os.path.join(userpath, entry.name)
                    name = entry.name.lower()
                    list_files = os.listdir(f_path)
                    print(f_path)

                    j = 0
                    for filename in list_files: 
                        file, file_extension = os.path.splitext(filename)
                        dst = "{}_{}{}".format(name, j, file_extension.lower())
                                         
                        while dst in list_files:
                            list_files.remove(dst)
                            print('such file already exists: ', dst)
                            j+=1
                            dst = "{}_{}{}".format(name, j, file_extension.lower())
                                    
                        else:
                            if len(list_files)!=0:
                                print(filename, '--->\t', dst)
                                os.rename(os.path.join(f_path,filename), os.path.join(f_path,dst)) 
                            j+=1

    except Exception as e:
        print(str(e))

main()

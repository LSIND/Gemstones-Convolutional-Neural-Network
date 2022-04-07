#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

def main():
    userpath = sys.argv[1]
    try:
        if(os.path.isdir(userpath)):        
            with os.scandir(userpath) as rootfolder:
                for entry in rootfolder:
                    if(entry.is_dir()):
                        f_path = os.path.join(userpath, entry.name)                        
                        name = entry.name.lower()
                        listfiles = os.listdir(f_path)
                        #print(listfiles)
                        
                        j = 0
                        dst_filenames = []
                        if (len(listfiles) > 0):
                            print(f_path, ':', len(listfiles))
                            for filename in listfiles:
                                file, file_ext = os.path.splitext(filename)
                                dst_filename = f'{name}_{j}{file_ext.lower()}'
                                dst_filenames.append(dst_filename)
                                j+=1
                            
                            #пересечение множеств
                            inter = set(dst_filenames)& set(listfiles)
                            print(inter)
                            
                            dst_filenamesF = [el for el in dst_filenames if el not in inter]
                            listfilesF = [el for el in listfiles if el not in inter]
                            
                            finalNames = list(zip(listfilesF, dst_filenamesF))
                            print('To rename:', len(finalNames))
                            
                            for filename in finalNames:                                
                                #print(os.path.join(f_path, filename[0]))
                                #print(os.path.join(f_path, filename[1]))
                                os.rename(os.path.join(f_path, filename[0]), os.path.join(f_path, filename[1]))

        else:
            print(f'error: {userpath} incorrect')

    except Exception as e:
        print(str(e))

main()

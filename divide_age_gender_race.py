# -*- coding:utf-8 -*-
from random import shuffle

# black, white, Asian, Hispanic, and other races - Morph
# White and Black - UTK
# Block-0, White-1
import numpy as np
import os
import easydict

FLAGS = easydict.EasyDict({"Morph_txt": "D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/Morph/All/age_gender_race.txt",
                           
                           "Morph_img": "D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/Morph/All/Crop_dlib/",
                           
                           "UTK_img": "D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/UTK/UTKFace/",

                           "UTK_txt": "D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/UTK/age_gender_race.txt",
                           
                           "AFAD_txt": "D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/AFAD/age_gender_race.txt",
                           
                           "AFAD_img": "D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/AFAD/fix_AFAD/",
                           
                           "AAF_txt": "D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/AAF/crop/age_gender_race.txt",
                           
                           "AAF_img": "D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/AAF/crop/merge_img/"})

# 모든 학습 데이터를 동일한 개수로 맞추어야하고, 테스트 개수역시 다 동일하게 맞추어야 한다(WM, WF, BM, BF, AM, AF 개수 모두 동일).
def main():

    # Morph 16 ~ 63
    # UTK 16 ~ 63
    # AFAD 16 ~ 63
    # AAF 16 ~ 63

    all_Morph_name = np.loadtxt(FLAGS.Morph_txt, dtype="<U100", skiprows=0, usecols=[0, 1, 2, 3])
    all_Morph_img = []
    for i in range(len(all_Morph_name)):
        data_list = all_Morph_name[i]
        all_Morph_img.append(FLAGS.Morph_img + data_list[0])

    all_UTK_name = np.loadtxt(FLAGS.UTK_txt, dtype="<U100", skiprows=0, usecols=[0, 1, 2, 3])
    all_UTK_img = []
    for i in range(len(all_UTK_name)):
        data_list = all_UTK_name[i]
        all_UTK_img.append(FLAGS.UTK_img + data_list[0])

    all_AFAD_name = np.loadtxt(FLAGS.AFAD_txt, dtype="<U100", skiprows=0, usecols=[0, 1, 2, 3])
    all_AFAD_img = []
    for i in range(len(all_AFAD_name)):
        data_list = all_AFAD_name[i]
        all_AFAD_img.append(FLAGS.AFAD_img + data_list[0])

    all_AAF_name = np.loadtxt(FLAGS.AAF_txt, dtype="<U100", skiprows=0, usecols=[0, 1, 2, 3])
    all_AAF_img = []
    for i in range(len(all_AAF_name)):
        data_list = all_AAF_name[i]
        all_AAF_img.append(FLAGS.AAF_img + data_list[0])

    age_list = np.arange(16, 64, dtype=np.int32)
    write_Morph_tr = open("D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/train_data/Morph/train.txt", "w")
    write_UTK_tr = open("D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/train_data/UTK/train.txt", "w")
    write_AFAD_tr = open("D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/train_data/AFAD/train.txt", "w")
    write_AAF_tr = open("D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/train_data/AAF/train.txt", "w")

    write_Morph_te = open("D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/train_data/Morph/test.txt", "w")
    write_UTK_te = open("D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/train_data/UTK/test.txt", "w")
    write_AFAD_te = open("D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/train_data/AFAD/test.txt", "w")
    write_AAF_te = open("D:/[1]DB/[5]4th_paper_DB/age_estimation_banchmark/train_data/AAF/test.txt", "w")

    for i in range(16, 64):
        Morph_age = []
        Morph_gender = []
        Morph_race = []
        for j in range(len(all_Morph_name)):
            if i == int(all_Morph_name[j][1]):
                Morph_age.append(all_Morph_img[j])
                Morph_gender.append(all_Morph_name[j][2])
                Morph_race.append(all_Morph_name[j][3])
        A = list(zip(Morph_age, Morph_gender, Morph_race))
        shuffle(A)
        Morph_age, Morph_gender, Morph_race = zip(*A)
        Morph_age_tr = Morph_age[:len(Morph_age)//2]
        Morph_age_te = Morph_age[len(Morph_age)//2:]
        Morph_gender_tr = Morph_gender[:len(Morph_age)//2]
        Morph_gender_te = Morph_gender[len(Morph_age)//2:]
        Morph_race_tr = Morph_race[:len(Morph_age)//2]
        Morph_race_te = Morph_race[len(Morph_age)//2:]
        print(len(Morph_age_tr))

        UTK_age = []
        UTK_gender = []
        UTK_race = []
        for j in range(len(all_UTK_name)):
            if i == int(all_UTK_name[j][1]):
                UTK_age.append(all_UTK_img[j])
                UTK_gender.append(all_UTK_name[j][2])
                UTK_race.append(all_UTK_name[j][3])
        B = list(zip(UTK_age, UTK_gender, UTK_race))
        shuffle(B)
        UTK_age, UTK_gender, UTK_race = zip(*B)
        UTK_age_tr = UTK_age[:len(UTK_age)//2]
        UTK_age_te = UTK_age[len(UTK_age)//2:]
        UTK_gender_tr = UTK_gender[:len(UTK_age)//2]
        UTK_gender_te = UTK_gender[len(UTK_age)//2:]
        UTK_race_tr = UTK_race[:len(UTK_age)//2]
        UTK_race_te = UTK_race[len(UTK_age)//2:]
        print(len(UTK_age_tr))

        AFAD_age = []
        AFAD_gender = []
        AFAD_race = []
        for j in range(len(all_AFAD_name)):
            if i == int(all_AFAD_name[j][1]):
                AFAD_age.append(all_AFAD_img[j])
                AFAD_gender.append(all_AFAD_name[j][2])
                AFAD_race.append(all_AFAD_name[j][3])
        C = list(zip(AFAD_age, AFAD_gender, AFAD_race))
        shuffle(C)
        AFAD_age, AFAD_gender, AFAD_race = zip(*C)
        AFAD_age_tr = AFAD_age[:len(AFAD_age)//2]
        AFAD_age_te = AFAD_age[len(AFAD_age)//2:]
        AFAD_gender_tr = AFAD_gender[:len(AFAD_age)//2]
        AFAD_gender_te = AFAD_gender[len(AFAD_age)//2:]
        AFAD_race_tr = AFAD_race[:len(AFAD_age)//2]
        AFAD_race_te = AFAD_race[len(AFAD_age)//2:]
        print(len(AFAD_age_tr))

        AAF_age = []
        AAF_gender = []
        AAF_race = []
        for j in range(len(all_AAF_name)):
            if i == int(all_AAF_name[j][1]):
                AAF_age.append(all_AAF_img[j])
                AAF_gender.append(all_AAF_name[j][2])
                AAF_race.append(all_AAF_name[j][3])
        D = list(zip(AAF_age, AAF_gender, AAF_race))
        shuffle(D)
        AAF_age, AAF_gender, AAF_race = zip(*D)
        AAF_age_tr = AAF_age[:len(AAF_age)//2]
        AAF_age_te = AAF_age[len(AAF_age)//2:]
        AAF_gender_tr = AAF_gender[:len(AAF_age)//2]
        AAF_gender_te = AAF_gender[len(AAF_age)//2:]
        AAF_race_tr = AAF_race[:len(AAF_age)//2]
        AAF_race_te = AAF_race[len(AAF_age)//2:]
        print(len(AAF_age_tr))

        min_tr_data = min(min(len(Morph_age_tr), len(UTK_age_tr)), min(len(AFAD_age_tr), len(AAF_age_tr)))
        min_te_data = min(min(len(Morph_age_te), len(UTK_age_te)), min(len(AFAD_age_te), len(AAF_age_te)))

        ####################################################################################################
        Morph_age_tr = Morph_age_tr[:min_tr_data]
        UTK_age_tr = UTK_age_tr[:min_tr_data]
        AFAD_age_tr = AFAD_age_tr[:min_tr_data]
        AAF_age_tr = AAF_age_tr[:min_tr_data]

        Morph_age_te = Morph_age_te[:min_tr_data]
        UTK_age_te = UTK_age_te[:min_tr_data]
        AFAD_age_te = AFAD_age_te[:min_tr_data]
        AAF_age_te = AAF_age_te[:min_tr_data]

        Morph_gender_tr = Morph_gender_tr[:min_tr_data]
        UTK_gender_tr = UTK_gender_tr[:min_tr_data]
        AFAD_gender_tr = AFAD_gender_tr[:min_tr_data]
        AAF_gender_tr = AAF_gender_tr[:min_tr_data]

        Morph_gender_te = Morph_gender_te[:min_tr_data]
        UTK_gender_te = UTK_gender_te[:min_tr_data]
        AFAD_gender_te = AFAD_gender_te[:min_tr_data]
        AAF_gender_te = AAF_gender_te[:min_tr_data]

        Morph_race_tr = Morph_race_tr[:min_tr_data]
        UTK_race_tr = UTK_race_tr[:min_tr_data]
        AFAD_race_tr = AFAD_race_tr[:min_tr_data]
        AAF_race_tr = AAF_race_tr[:min_tr_data]

        Morph_race_te = Morph_race_te[:min_tr_data]
        UTK_race_te = UTK_race_te[:min_tr_data]
        AFAD_race_te = AFAD_race_te[:min_tr_data]
        AAF_race_te = AAF_race_te[:min_tr_data]

        for k in range(len(Morph_age_tr)):
            write_Morph_tr.write(Morph_age_tr[k].split('/')[-1])
            write_Morph_tr.write(" ")
            write_Morph_tr.write(str(i))
            write_Morph_tr.write(" ")
            write_Morph_tr.write(Morph_gender_tr[k])
            write_Morph_tr.write(" ")
            write_Morph_tr.write(Morph_race_tr[k])
            write_Morph_tr.write("\n")
            write_Morph_tr.flush()

            write_Morph_te.write(Morph_age_te[k].split('/')[-1])
            write_Morph_te.write(" ")
            write_Morph_te.write(str(i))
            write_Morph_te.write(" ")
            write_Morph_te.write(Morph_gender_te[k])
            write_Morph_te.write(" ")
            write_Morph_te.write(Morph_race_te[k])
            write_Morph_te.write("\n")
            write_Morph_te.flush()

        for k in range(len(UTK_age_tr)):
            write_UTK_tr.write(UTK_age_tr[k].split('/')[-1])
            write_UTK_tr.write(" ")
            write_UTK_tr.write(str(i))
            write_UTK_tr.write(" ")
            write_UTK_tr.write(UTK_gender_tr[k])
            write_UTK_tr.write(" ")
            write_UTK_tr.write(UTK_race_tr[k])
            write_UTK_tr.write("\n")
            write_UTK_tr.flush()

            write_UTK_te.write(UTK_age_te[k].split('/')[-1])
            write_UTK_te.write(" ")
            write_UTK_te.write(str(i))
            write_UTK_te.write(" ")
            write_UTK_te.write(UTK_gender_te[k])
            write_UTK_te.write(" ")
            write_UTK_te.write(UTK_race_te[k])
            write_UTK_te.write("\n")
            write_UTK_te.flush()

        for k in range(len(AFAD_age_tr)):
            write_AFAD_tr.write(AFAD_age_tr[k].split('/')[-1])
            write_AFAD_tr.write(" ")
            write_AFAD_tr.write(str(i))
            write_AFAD_tr.write(" ")
            write_AFAD_tr.write(AFAD_gender_tr[k])
            write_AFAD_tr.write(" ")
            write_AFAD_tr.write(AFAD_race_tr[k])
            write_AFAD_tr.write("\n")
            write_AFAD_tr.flush()

            write_AFAD_te.write(AFAD_age_te[k].split('/')[-1])
            write_AFAD_te.write(" ")
            write_AFAD_te.write(str(i))
            write_AFAD_te.write(" ")
            write_AFAD_te.write(AFAD_gender_te[k])
            write_AFAD_te.write(" ")
            write_AFAD_te.write(AFAD_race_te[k])
            write_AFAD_te.write("\n")
            write_AFAD_te.flush()

        for k in range(len(AAF_age_tr)):
            write_AAF_tr.write(AAF_age_tr[k].split('/')[-1])
            write_AAF_tr.write(" ")
            write_AAF_tr.write(str(i))
            write_AAF_tr.write(" ")
            write_AAF_tr.write(AAF_gender_tr[k])
            write_AAF_tr.write(" ")
            write_AAF_tr.write(AAF_race_tr[k])
            write_AAF_tr.write("\n")
            write_AAF_tr.flush()

            write_AAF_te.write(AAF_age_te[k].split('/')[-1])
            write_AAF_te.write(" ")
            write_AAF_te.write(str(i))
            write_AAF_te.write(" ")
            write_AAF_te.write(AAF_gender_te[k])
            write_AAF_te.write(" ")
            write_AAF_te.write(AAF_race_te[k])
            write_AAF_te.write("\n")
            write_AAF_te.flush()

if __name__ == "__main__":
    main()

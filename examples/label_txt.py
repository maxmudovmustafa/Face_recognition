# function for saving coordinates of detected faces in images as txt file
def label_txt(pathdr, lab_dir): # pathdr is our training photos folder, lab_dir is where this labels text will be saved
    for fol in os.listdir(pathdr): # for each folder (person) in train/test directory:
        tfile = open(lab_dir+fol+".txt","w+") # note that lab_dir must exist
        for img in os.listdir(pathdr+fol): # for each image in folder (one person):
            pathimg=os.path.join(pathdr+fol, img)
            #print(pathimg)
            pic=cv2.imread(pathimg)
            x1, y1, x2, y2=face_dnn(pic, True) # face detection and then saving into txt file
            tfile.write(img+' '+str(x1)+' '+str(x2)+' '+str(y1)+' '+str(y2)+'\n')
        tfile.close()
print('Saved')

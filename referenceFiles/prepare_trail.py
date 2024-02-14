f=open("b.txt","r")
f=f.readlines()
f=[i.rstrip("\n") for i in f]
f1=open("test.txt","w")
for i in f:
    k=i.split("/")
    spk=k[-1].split("_")[0]
    wavfile=k[-2]+"/"+k[-1]
    for j in f:
        k1=j.split("/")
        spk1=k1[-1].split("_")[0]
        wavfile1=k1[-2]+"/"+k1[-1]
        if spk==spk1:
            f1.write("%s %s %s\n"%("1",wavfile,wavfile1))
        else:
            f1.write("%s %s %s\n"%("0",wavfile,wavfile1))
            
        

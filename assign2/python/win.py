w = []

# orthogonal rows 48
for i in range(4) :
	for j in range(4) :
		r1, r2, r3, r4, r5, r6 = [], [], [], [], [], []
		for k in range(4) :
			r1.append((i*4+j)*4+k)
                        r2.append((i*4+k)*4+j)
                        r3.append((j*4+i)*4+k)
                        r4.append((j*4+k)*4+i)
                        r5.append((k*4+i)*4+j)
                        r6.append((k*4+j)*4+i)
		if not r1 in w : w.append(r1)
		if not r2 in w : w.append(r2)
                if not r3 in w : w.append(r3)
                if not r4 in w : w.append(r4)
                if not r5 in w : w.append(r5)
                if not r6 in w : w.append(r6)

print len(w),"orthogoanl 48"

# main diagonals 4
NumRows=4
r = []
for i in range(4) :
	r.append((4*i+i)*4+i)
w.append(r)
r = []
for i in range(4) :
        r.append((i*NumRows+(NumRows-i-1))*NumRows+i)
w.append(r)
r = []
for i in range(4) :
        r.append((i*NumRows+i)*NumRows+(NumRows-i-1))
w.append(r)
r = []
for i in range(4) :
        r.append((NumRows*i+(NumRows-i-1))*NumRows+(NumRows-i-1))
w.append(r)

print len(w)-48,"main diagonal 4"

# diagonal rows 24
for i in range(4) :
	r1, r2 = [], []
        for j in range(4) :
                for k in range(4) :
			if j == k : r1.append((i*4+j)*4+k)
			if j+k == 3 : r2.append((i*4+j)*4+k)
        if not r1 in w : w.append(r1)
        if not r2 in w : w.append(r2)

for i in range(4) :
	r = []
	for j in range(4) :
		r.append(i+j*20)
	if not r in w : w.append(r)

for i in range(4) :
        r = []
        for j in range(4) :
                r.append(12+i+j*12)
        if not r in w : w.append(r)

for i in range(4) :
        r = []
        for j in range(4) :
                r.append(i*4+j*17)
        if not r in w : w.append(r)

for i in range(4) :
        r = []
        for j in range(4) :
                r.append(3+i*4+j*15)
        if not r in w : w.append(r)


print len(w)-48-4,"diagonal rows 24"

for i in range(len(w)) :
	print w[i],len(w[i])
print len(w),3*'\n'

txt = "{\n"
for i in range(len(w)) :
	txt += "{"
	for j in range(len(w[i])) :
		txt += str(w[i][j]) + ",\t"
	txt = txt[:-2] + "},\n"
txt += "};"

print txt

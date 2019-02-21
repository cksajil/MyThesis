import pandas as pd
import matplotlib.pyplot as plt

#

width = 4
height = width / 1.618
labelsize = 10
legendfont = 7
lwidth = 0.8

plt.rc('pdf', fonttype=42)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=labelsize)
plt.rc('ytick', labelsize=labelsize)
plt.rc('axes', labelsize=labelsize)

#

data = pd.read_csv(open('Results/RakeMVDRRoomL.csv', 'rb'))
# data 	= 	pd.read_csv\(open('Results/MVDRDirect4by6.csv','rb'))

fig, ax = plt.subplots()
fig.subplots_adjust(left=.2, bottom=.2, right=.97, top=.90)
# plt.plot(data['Alpha'], data['ANCRed'],c ='k', label = 'ANCRed')
# plt.plot(data['Alpha'], data['ANCRedBeam'],c ='b', label =
# 'ANCRedBeam')
plt.plot(
    data['Alpha'],
    data['ANCRed'],
    c='b',
    label='ANCRed',
    linestyle='solid')
plt.plot(
    data['Alpha'],
    data['ANCRedBeam'],
    c='k',
    label='ANCRedBeam',
    linestyle='dashed')
plt.plot(
    data['Alpha'],
    data['ANCRedBeam'] -
    data['ANCRed'],
    c='r',
    label='improvement',
    linestyle='dotted')
# plt.plot(data['Alpha'], data['EAdB'],c ='b', label = 'EAdB')

print data.describe()

plt.xlabel('Absorption Coefficient ($\\alpha$)')
plt.ylabel('Noise Reduction (dB)')
plt.legend()
plt.show()

fig.set_size_inches(width, height)
fig.savefig('Images/MVDR_Polyhedral.png', dpi=600)
# fig.savefig('Images/MVDRDirect4by6.png', dpi = 600)
plt.close()

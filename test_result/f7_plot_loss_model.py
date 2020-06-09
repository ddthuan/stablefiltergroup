# https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting/33213196

from libcore import *
from matplotlib.ticker import ScalarFormatter

x = np.array([0.])
y = np.array([0.])
    
fig = plt.figure()
ax1  = fig.add_subplot(111)
ax1.set_xlim([1, 30], )
ax1.set_ylim([1, 800])

# =============================================================================
# line_raw, = ax1.plot(x, y, 'b-')
# line_complex, = ax1.plot(x, y, 'r-') 
# =============================================================================

line_raw, = ax1.loglog(x, y, 'b-', basey=2, label='Standard Model')
line_complex, = ax1.loglog(x, y, 'r-', basey=2, label='Complex Model')

formatter = ScalarFormatter()
formatter.set_scientific(False)
ax1.xaxis.set_major_formatter(formatter)
ax1.yaxis.set_major_formatter(formatter)

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color ='b')
ax1.tick_params('y',   colors='b')

fig.tight_layout()

df_01_raw = pd.read_csv('../results/loss/loss_cifar1_raw.csv')
df_02_complex = pd.read_csv('../results/loss/loss_cifar1_complex.csv')

epoch = df_01_raw["epoch"]
loss_raw = df_01_raw["loss"]
loss_complex = df_02_complex["loss"]

#for i in range(len(epoch)):
for i in range(30):
    xd = line_raw.get_xdata() ; yd = line_raw.get_ydata()
    xd = np.append(xd, i)
    yd = np.append(yd, loss_raw[i])
    line_raw.set_xdata(xd)    ; line_raw.set_ydata(yd)
    fig.canvas.draw()
    
    xd = line_complex.get_xdata() ; yd = line_complex.get_ydata()
    xd = np.append(xd, i)
    yd = np.append(yd, loss_complex[i])
    line_complex.set_xdata(xd)    ; line_complex.set_ydata(yd)
    fig.canvas.draw()

ax1.legend()
plt.show()    
fig.savefig('metric/loss_inverse_cifar.png', dpi=160, bbox_inches="tight")
    
    

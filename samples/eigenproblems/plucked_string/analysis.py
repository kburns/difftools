"""
Test eigenproblem: frequencies of a plucked string.

Solve

    u_xx = - L**2 * u

    u(-1) = 0
    u(1) = 0

"""


import numpy as np
import matplotlib.pyplot as plt
import plucked_string


# Solve at two resolutions
s1 = 16
s2 = 32
(L1, ev1, ef1) = plucked_string.main(s1)
(L2, ev2, ef2) = plucked_string.main(s2)

# Compare to expected eigenvalues
ex1 = -(np.arange(1, ev1.size+1) * np.pi / 2.) ** 2
ex2 = -(np.arange(1, ev2.size+1) * np.pi / 2.) ** 2
err1 = np.abs(ev1 - ex1)
err2 = np.abs(ev2 - ex2)

# Plot absolute eigenvalue errors
plt.figure(1, figsize=(7,5))
plt.clf()

plt.axhline(0.01, ls='dashed', c='k')
plt.semilogy(np.arange(1, ev1.size+1), err1, 'o-k', label='n=%i' %s1)
plt.semilogy(np.arange(1, ev2.size+1), err2, 'o-k', mfc='w', mew=1, label='n=%i' %s2)
plt.legend(loc='lower right', fontsize='small')
plt.xlabel(r'Mode number $j$')
plt.ylabel(r'$j$-th eigvenvalue abs. error')
plt.savefig('abs_error.png', dpi=200)

# Plot good and bad eigenmode approximations
(fig, (ax1, ax2)) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7,5))
x = np.linspace(-1., 1., 1000)

m = 4
ynum = -ef1[m].evaluate(x)
ynum /= ynum.max()
yex = np.cos(np.sqrt(-ex1[m]) * x)
ax1.plot(x, ynum, '-k', label='Numerical')
ax1.plot(x, yex, '-b', lw=4, alpha=0.3, label='Exact')
ax1.set_ylim([-1.2, 1.2])
ax1.set_ylabel('5th mode')
ax1.legend(loc='lower right', fontsize='small')

m = 11
ynum = -ef1[m].evaluate(x)
ynum /= ynum.max()
yex = np.sin(np.sqrt(-ex1[m]) * x)
ax2.plot(x, ynum, '-k', label='Numerical')
ax2.plot(x, yex, '-b', lw=4, alpha=0.3, label='Exact')
ax2.set_ylim([-1.2, 1.2])
ax2.set_xlabel(r'$x$')
ax2.set_ylabel('12th mode')

plt.savefig('eigenmodes.png', dpi=200)


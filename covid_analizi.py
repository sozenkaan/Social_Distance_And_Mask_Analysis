import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DAYS = 180
POPULATION = 100000
SPREAD_FACTOR = 0.2
DAYS_TO_RECOVER = 10
INITIALLY_AFFECTED = 4
city = pd.DataFrame(data={'id': np.arange(POPULATION), 'infected': False, 'recovery_day': None, 'recovered': False})
city = city.set_index('id')

firstCases = city.sample(INITIALLY_AFFECTED, replace=False)
city.loc[firstCases.index, 'infected'] = True
city.loc[firstCases.index, 'recovery_day'] = DAYS_TO_RECOVER

stat_active_cases = [INITIALLY_AFFECTED]
stat_recovered = [0]

for today in range(1, DAYS):
    # Mark people who have recovered today
    city.loc[city['recovery_day'] == today, 'recovered'] = True
    city.loc[city['recovery_day'] == today, 'infected'] = False

    # Calcuate the number of people who are infected today
    spreadingPeople = city[(city['infected'] == True)]
    totalCasesToday = round(len(spreadingPeople) * SPREAD_FACTOR)
    casesToday = city.sample(totalCasesToday, replace=True)
    # Ignore people who were already infected in casesToday
    casesToday = casesToday[(casesToday['infected'] == False) & (casesToday['recovered'] == False)]
    # Mark the new cases as infected, and their recovery day
    city.loc[casesToday.index, 'infected'] = True
    city.loc[casesToday.index, 'recovery_day'] = today + DAYS_TO_RECOVER

    stat_active_cases.append(len(city[city['infected'] == True]))
    # stat_recovered.append(len(city[city['recovered'] == True]))

    # Try and reduce the SPREAD_FACTOR to simulate the effects of different levels of social distancing
    # if today >= 5:
    #   SPREAD_FACTOR = 1
    # if today >= 10:
    #   SPREAD_FACTOR = 0.1

import matplotlib.pyplot as plt

plt.bar(x=np.arange(DAYS), height=stat_active_cases, color="red")
plt.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Burada km ve seir modelleri kullanılarak sosyal mesafe verileri
# kullanılarak analiz gerçekleştirilmiştir.


import numpy as np
import matplotlib.pyplot as plt


def km_model(init_vals, params, t):
    x_0, y_0, z_0 = init_vals
    x, y, z = [x_0], [y_0], [z_0]
    l, k = params
    dt = t[2] - t[1]  # Assumes constant time steps
    for t_ in t[:-1]:
        next_x = x[-1] - (k * x[-1] * y[-1]) * dt
        next_y = y[-1] + (k * x[-1] * y[-1] - l * y[-1]) * dt
        next_z = z[-1] + (l * next_y) * dt
        x.append(next_x)
        y.append(next_y)
        z.append(next_z)

    return np.stack([x, y, z]).T


def base_seir_model(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta * S[-1] * I[-1]) * dt
        next_E = E[-1] + (beta * S[-1] * I[-1] - alpha * E[-1]) * dt
        next_I = I[-1] + (alpha * E[-1] - gamma * I[-1]) * dt
        next_R = R[-1] + (gamma * I[-1]) * dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T


# Define parameters
t_max = 100
dt = .9
t = np.linspace(0, t_max, int(t_max / dt) + 1)
N = 10000
init_vals = 1 - 1 / N, 1 / N, 0, 0
alpha = 0.2
beta = 1.75
gamma = 0.5
params = alpha, beta, gamma
# Run simulation
results = base_seir_model(init_vals, params, t)
plt.figure(figsize=(12, 8))
plt.plot(results)
plt.legend(['Exposed', 'Infected'])
plt.xlabel('Time(Days)')
plt.show()

init_vals = [999, 1, 0]
params = [1e-1, 1e-3]
t_max = 100
dt = 0.1
t = np.linspace(0, t_max, int(t_max / dt))
km_results = km_model(init_vals, params, t)
# Plot results
plt.figure(figsize=(12, 8))
plt.plot(km_results)
plt.legend(['Susceptible', 'Sick', 'Recovered'])
plt.xlabel('Time Steps')
plt.show()

params = [5e-1, 1e-3]
km_results = km_model(init_vals, params, t)
plt.figure(figsize=(12, 8))
plt.plot(km_results)
plt.legend(['Susceptible', 'Sick', 'Recovered'])
plt.xlabel('Time Steps')
plt.title(r'KM Model with $l={}$'.format(params[0]))
plt.show()

params = [1e-1, 1e-2]
km_results = km_model(init_vals, params, t)
plt.figure(figsize=(12, 8))
plt.plot(km_results)
plt.legend(['Susceptible', 'Sick', 'Recovered'])
plt.xlabel('Time Steps')
plt.title(r'KM Model with $k={}$'.format(params[1]))
plt.show()

params = [1e-1, 1e-4]
km_results = km_model(init_vals, params, t)
plt.figure(figsize=(12, 8))
plt.plot(km_results)
plt.legend(['Susceptible', 'Sick', 'Recovered'])
plt.xlabel('Time Steps')
plt.title(r'KM Model with $k={}$'.format(params[1]))
plt.show()
print('R0 = {}'.format(sum(init_vals) * params[1] / params[0]))


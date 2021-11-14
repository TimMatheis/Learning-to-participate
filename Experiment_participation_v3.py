#%%
import numpy as np
import pandas as pd

def participationTrading(numberGames):

    index = 0

    # Saving df
    stats = pd.DataFrame(columns=['run', 'episode', r'$m$', r'$n$', r'$\theta^1$', r'$\theta^2$', r'Price $m$', r'Price $n$', 'acc_rewards']) # more, others

    # Multiple runs
    run = 1
    number_runs = 20

    while run <= number_runs:

        # Define variables
        theta_1 = 0.5
        theta_2 = 0.5

        m = 0
        n = 0

        m_update = 0.02
        n_update = 0.02

        alpha = 0.02
        beta = 0.1
        gamma = 0.8

        acc_rewards_realized = 0

        # Loop to update
        for i in range (0, numberGames+1):

            price_m = (1/(1-gamma))* (-1)*( theta_1*theta_2 + theta_1 * (1-theta_2) *3 + (1-theta_1) * (1-theta_2) * 2 )
            price_n = (1/(1-gamma))* (-1)*( theta_1*theta_2 + (1 - theta_1) * theta_2 *3 + (1-theta_1) * (1-theta_2) * 2 )

            stats.loc[index] = [run, i, m, n, theta_1, theta_2, price_m, price_n, acc_rewards_realized] 
            index += 1

            # Probabilities
            p = [theta_1*theta_2, theta_1*(1-theta_2), (1-theta_1)*theta_2, (1-theta_1)*(1-theta_2)]

            # Minimum requirements have to be fulfilled: agents are only willing to trade if there is a non-negativ effect on their returns
            # Here the effect from trading on the immediate rewards is always exactly 0

            # Here beta does not change anything: beta*0 = 0, but if the reward sum is changed, beta alone would be sufficient and could be used insetead of m_update / n_update
            
            """
            min1_m = beta*(1/(1-gamma))*(price_m + theta_1*theta_2 + theta_1*(1-theta_2)*3 + (1-theta_1)*(1-theta_2)*2)
            min2_m = beta*(1/(1-gamma))*(price_m*(-1) -(theta_1*theta_2 + theta_1 *(1-theta_2)*3+(1-theta_1)*(1-theta_2)*2))
            """

            min1_m = beta * (price_m + (1/(1-gamma))*(theta_1*theta_2 + theta_1*(1-theta_2)*3 + (1-theta_1)*(1-theta_2)*2))
            min2_m = beta * (price_m*(-1) + (1/(1-gamma))*( -(theta_1*theta_2 + theta_1 *(1-theta_2)*3+(1-theta_1)*(1-theta_2)*2)))

            """
            min1_n = beta*(1/(1-gamma))*(price_n + theta_1*theta_2 + (1 - theta_1) * theta_2 *3 + (1-theta_1)*(1-theta_2)*2)
            min2_n = beta*(1/(1-gamma))*(price_n*(-1) -(theta_1*theta_2 + (1 - theta_1) * theta_2 * 3 + (1-theta_1)*(1-theta_2)*2))
            """

            min1_n = beta * (price_n + (1/(1-gamma))*(theta_1*theta_2 + (1 - theta_1) * theta_2 *3 + (1-theta_1)*(1-theta_2)*2))
            min2_n = beta * (price_n*(-1) + (1/(1-gamma))*(-(theta_1*theta_2 + (1 - theta_1) * theta_2 * 3 + (1-theta_1)*(1-theta_2)*2)))


            if( min(min1_m, min2_m) >= 0 ):
                m = min( m + m_update, 0.5 )

            if( min(min1_n, min2_n) >= 0 ):
                n = min( n + n_update, 0.5 )

            # Probabilities cannot be greater 1 or smaller 0
            theta_1 = min( max(theta_1 + alpha/(1-gamma)*(2*n+m-1), 0), 1 )
            theta_2 = min( max(theta_2 + alpha/(1-gamma)*(2*m+n-1), 0), 1 )


            # Rewards: first for player 1, then for player 2
            rewards = [ [-1*(1-m)+(-1)*n, -1*(1-n)+(-1)*m], [-3*(1-m)+0*n, 0*(1-n)+(-3)*m],
                        [0*(1-m)+(-3)*n, (-3)*(1-n)+0*m], [-2*(1-m)+(-2)*n, -2*(1-n)+(-2)*m] ]

            # choice = np.random.choice([0,1,2,3], size=1, p=p)[0]
            # rewards_realized_1 = rewards[choice][0]
            # rewards_realized_2 = rewards[choice][1]
            # check_acc = sum([rewards_realized_1, rewards_realized_2])
            acc_rewards_realized = np.random.choice([sum(rewards[0]), sum(rewards[1]), sum(rewards[2]), sum(rewards[3])], size=1, p=p)[0]

            if ( (i%3) == 1 ):
                print("Game number: ", i)
                print("min1_n: ", min1_n)
                print("Theta 1: {:.2f}, Theta 2: {:.2f}".format(theta_1, theta_2))
                print("m: {:.2f}, n: {:.2f}".format(m, n))
                print("Price m: {:.2f}, price_n: {:.2f}".format(price_m, price_n))
                print("Probabilities: ", p)
                print("Rewards: ", rewards)
                # print(" Acc. rewards realized 1: ", rewards_realized_1)
                # print(" Acc. rewards realized 2: ", rewards_realized_2)
                print(" Acc. rewards realized: ", acc_rewards_realized)
                # print(" Check acc. rewards realized: ", check_acc)
            
        run += 1

            # stats.loc[i] = [i, m, n, theta_1, theta_2, price_m, price_n, acc_rewards_realized] 

    stats.to_csv('2-player-with-penalization_test_run_multiple.csv')

participationTrading(100)

# %%

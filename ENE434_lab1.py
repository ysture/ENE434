
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

oil_fields = pd.read_csv("http://jmaurit.github.io/analytics/labs/data/oil_fields_cross.csv")
oil_fields = oil_fields.dropna()
oil_fields['producing_from']= pd.to_datetime(oil_fields['producing_from'])  # converting dates to datetime
oil_fields.head()

'''
Question 1:
Create a variable invest_per_rec which is investment per million sm3 in recoverable oil (recoverable_oil). 
Plot these variables against producing_from variable and the original recoverable_oil variable. 
How do you interpret the results?
'''

oil_fields.loc[:,'invest_per_rec'] = round(oil_fields['total.invest'] / oil_fields['recoverable_oil'] ,5)
pd.set_option('display.max_rows', 1000)
oil_fields

# plotting 'producing_from' against 'recoverable_oil'
plt.scatter(oil_fields['producing_from'], oil_fields['recoverable_oil'])
plt.ylabel('Investment per recoverable sm3 of oil')
plt.xlabel('Field produced from')
plt.show()

'''
Question 2:
Create a list of the 5 “cheapest” oil fields, that is where the investment is lowest per recoverable oil. 
What do these tend to have in common?
'''
# Consider to use oil_fields[oil_fields['invest_per_rec'] != np.inf] instead of oil_fields
pd.set_option('display.max_columns', 5)
oil_fields.sort_values(by='invest_per_rec').head()

'''
Question 3:
I have a hypothesis that oil fields farther north are more expensive to exploit. 
Explore this hypothesis. Do you think it has merit?
'''
plt.scatter(oil_fields['lat'], oil_fields['invest_per_rec'])
plt.ylabel('Investment per recoverable sm3 of oil')
plt.xlabel('Latitude')
plt.show()

'''
Question 4:
Open-ended question: Accessing and importing data

Actually finding and accessing interesting data you want can be challenging. Importing it into R into the correct format can also be challenging. Here you get a taste of this.

a)
Go to the data portion of the Norwegian Petrioleum Directorate
'''
inv = pd.read_csv()

'''
b)
The tabs at the top indicate the different types of data that is available by level/theme. Try to find some interesting dataset and download it as a .csv file. (Hint, on the left-hand panel, go down to “table view”, then you get a table of data, which you can export by clicking on “Export CSV”).
'''

'''
c)
Once you have downloaded the data, import the data into r using the read_csv() command.
'''

'''
d)
If there is a date variable, format that variable as a date (if read_csv() hasn’t automatically done so already)
'''

'''
e)
Plot the data in a meaningful way. Interpret the plot. Is there anything puzling about the data.
'''

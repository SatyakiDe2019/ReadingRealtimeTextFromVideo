#####################################################
#### Written By: SATYAKI DE                      ####
#### Written On: 22-Jul-2022                     ####
#### Modified On 25-Jul-2022                     ####
####                                             ####
#### Objective: This is the main calling         ####
#### python script that will invoke the          ####
#### clsReadingTextFromStream class to initiate  ####
#### the reading capability in real-time         ####
#### & display text via Web-CAM.                 ####
#####################################################

# We keep the setup code in a different class as shown below.
import clsReadingTextFromStream as rtfs

from clsConfig import clsConfig as cf

import datetime
import logging

###############################################
###           Global Section                ###
###############################################
# Instantiating all the main class

x1 = rtfs.clsReadingTextFromStream()

###############################################
###    End of Global Section                ###
###############################################

def main():
    try:
        # Other useful variables
        debugInd = 'Y'
        var = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        var1 = datetime.datetime.now()

        print('Start Time: ', str(var))
        # End of useful variables

        # Initiating Log Class
        general_log_path = str(cf.conf['LOG_PATH'])

        # Enabling Logging Info
        logging.basicConfig(filename=general_log_path + 'readingTextFromVideo.log', level=logging.INFO)

        print('Started reading text from videos!')

        # Execute all the pass
        r1 = x1.processStream(debugInd, var)

        if (r1 == 0):
            print('Successfully read text from the Live Stream!')
        else:
            print('Failed to read text from the Live Stream!')

        var2 = datetime.datetime.now()

        c = var2 - var1
        minutes = c.total_seconds() / 60
        print('Total difference in minutes: ', str(minutes))

        print('End Time: ', str(var1))

    except Exception as e:
        x = str(e)
        print('Error: ', x)

if __name__ == "__main__":
    main()

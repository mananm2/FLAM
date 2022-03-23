import numpy as np
import tensorflow as tf


def federated_averaging(model,
                        SERVER_ROUNDS, LOCAL_EPOCHS, 
                        LOCAL_BATCH_SIZE,
                        LOCAL_LEARNING_RATE,
                        clientIDs, imageDict, segMaskDict,
                        testImages, testMasks):
    '''
    Executes the FedAvg algorithm.
    Performs local and server updates.
    Comments and print statements are self-explanatory.
    
    '''
    
    lossDict, testLoss = dict.fromkeys(clientIDs, np.array([])), []
    accuracyDict, testAccuracy = dict.fromkeys(clientIDs, np.array([])), []
    
    # Get the number of points in each client
    nk = [len(segMaskDict[clientID]) for clientID in clientIDs]
    # Compute the proportions of each client for weighted averaging
    proportionsDict = {clientIDs[i] : nk[i]/sum(nk) for i in range(len(clientIDs))}
    
    serverWeights = model.get_weights()

    for epoch in range(SERVER_ROUNDS):
        print('------ Server Epoch ' + str(epoch) + ' ------')

        clientWeights = {}
        # Client update
        for clientID in clientIDs:
            # Initialize a temporary client model of the same
            # architecture as the global model
            clientModel = tf.keras.models.clone_model(model)

            # Set weights equal to the current server state
            clientModel.set_weights(serverWeights)

            # Compile and run client model for LOCAL_EPOCHS with LOCAL_BATCH_SIZE
            print('Running local updates for ' + clientID + '...')
            clientModel.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LOCAL_LEARNING_RATE),
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                      metrics = ['accuracy'])
            history = clientModel.fit(imageDict[clientID], segMaskDict[clientID],
                      epochs = LOCAL_EPOCHS, batch_size = LOCAL_BATCH_SIZE,
                      shuffle = True)
            
            # Save the training loss and accuracy in corresponding dictionaries
            lossDict[clientID] = np.append(lossDict[clientID], list(history.history.values())[0])
            accuracyDict[clientID] = np.append(accuracyDict[clientID], list(history.history.values())[1])

            print('Saving local updates for ' + clientID + '...')
            # Save this client update in clientWeights
            clientWeights[clientID] = clientModel.get_weights()

        # Update server state
        print('Performing Server Update...')
        updatedServerWeights = []
        for j in range(len(serverWeights)):
            temp = np.zeros(serverWeights[j].shape)
            for clientID in clientIDs:
                temp += proportionsDict[clientID] * clientWeights[clientID][j]
            updatedServerWeights.append(temp)
        print('Done...')

        # Assign the averaged weights to the global model 
        print('Assigning current server state to the global model...')
        model.set_weights(updatedServerWeights)
        serverWeights = updatedServerWeights
        del updatedServerWeights # in case memory overloads

        # Keep track of test set performance
        print('Evaluating Test Set Performance...')
        tloss, tacc = model.evaluate(testImages, testMasks, verbose = 1)
        testLoss.append(tloss)
        testAccuracy.append(tacc)
        print('Done...\n')
        
    return model, serverWeights, lossDict, testLoss, accuracyDict, testAccuracy
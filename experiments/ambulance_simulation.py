import numpy as np
import pandas as pd
from datetime import datetime
import random  

# All time units in hour
# Global configs


def ambulance_simulation(base_loc_x, base_loc_y, T_max = 500, cool_off_period=10, num_ambulance=3, lambda_call=1, time_to_serve_avg=45./60,time_to_serve_std = 15./60,
travel_speed_when_on_call = 60., travel_speed_when_free = 40., seed = 1):

    global call_table
    global ambulance_table
    global simulation_table
    global ambulance
    
    ##
    ambulance = np.zeros((num_ambulance,6))
    ## column 0: time to finish last call, initiated to 0.
    ## column 1: x location of last call, initiated to 0.5
    ## column 2: y location of last call, initiated to 0.5
    ## column 3: x location of base.
    ## column 4: y location of base.
    ## column 5: status of ambulance
    #Creating call table to keep track of all call statuses
    call_table_columns = ['call_id','time_created', 'call_loc_x', 'call_loc_y', 'time_ambulance_assigned', 'time_ambulance_reached', 'time_completed', 'ambulance_id_assigned']
    call_table = pd.DataFrame(columns=call_table_columns)
    
    #Simulation event table
    simulation_table_columns = ['Current_time', 'Current_event', 'Time_of_next_call', 'Time_of_next_call_completed', 'Next_event', 'Next_event_time']
    simulation_table = pd.DataFrame(columns=simulation_table_columns)
    ##
    
    for j in range(0,num_ambulance):
        ambulance[j,1] = 0.5
        ambulance[j,2] = 0.5
        ambulance[j,3] = base_loc_x[j]
        ambulance[j,4] = base_loc_y[j]

    ## Other optional parameters: 
    # T_max = 500  # Number of hours.
    # cool_off_period = 10 #Remove calls recieved before this cool off period while reporting matrices. 
    # num_ambulance = 3
    # lambda_call = 1 # Avg calls per hour, used for poission distribution 
    # time_to_serve_avg = 45/60 #hours
    # time_to_serve_std = 15/60 #hours
    # travel_speed_when_on_call = 60 #in kmph, Vf
    # travel_speed_when_free = 40 #in kmph, Vs
    
    ## Checking for correct input parameters: 
    t1_all_def = datetime.now()
    
    len_x = np.size(base_loc_x)
    len_y = np.size(base_loc_y)
    
    if len_x != len_y:
        print("Number of X coordinates (",len_x,") ", "and Y coordinates(", len_y, ") ", "for base location, are not equal")
        
    if len_x != num_ambulance:
        print("Number of X coordinates (",len_x,") ", "and Number of ambulances(", num_ambulance, ") ", ", are not equal")
        
    ## Set global random state:
    random_state = np.random.RandomState(seed)
    
    ## MAIN CODE STARTS ###
    
    def time_to_serve_at_call_loc():
        global call_table
        global simulation_table
        mean = time_to_serve_avg #hours
        std = time_to_serve_std #hours
        shape = pow(mean/std,2)
        scale = pow(std,2)/mean
        t = random_state.gamma(shape, scale)  #t in hours
        return(t)
    #-- Unit testing done --#

    def travel_time(x1,y1,x2,y2,velocity):
        global call_table
        global simulation_table
        global ambulance
        #print("x2 inside travel_time function is:", x2)
        x_distance = abs(x2-x1)*30 #in kms
        #print("x_distance inside travel_time function is:", x_distance)
        y_distance = abs(y2-y1)*30 #in kms
        time = (x_distance + y_distance)/velocity
        
        return time

    #-- Unit testing done --#

    def interpolate(start_loc, end_loc, time_elapsed, total_time):
        global call_table
        global simulation_table

        # Handling total_time 0
        if total_time == 0:
            x = start_loc
        else:
            x = start_loc + (end_loc - start_loc)*time_elapsed/total_time
        return x
    #-- Unit testing done --#

    def current_location_of_free_ambulance(Current_time, ambulance_id):
        global call_table
        global simulation_table
        global ambulance
        
        time_since_last_free = Current_time - ambulance[ambulance_id,0] 
        time_to_reach_base = travel_time(ambulance[ambulance_id,1],ambulance[ambulance_id,2],ambulance[ambulance_id,3],ambulance[ambulance_id,4], travel_speed_when_free)
        
        if (ambulance[ambulance_id,0] == 0 or time_since_last_free > time_to_reach_base):
            current_location_x = ambulance[ambulance_id,3]
            current_location_y = ambulance[ambulance_id,4]
        else: 
            from_x = ambulance[ambulance_id,1]
            from_y = ambulance[ambulance_id,2]
            to_x = ambulance[ambulance_id,3]
            to_y = ambulance[ambulance_id,4]
            time_needed_to_cover_x_axis = travel_time(from_x,from_y,to_x,from_y,travel_speed_when_free)
            time_needed_to_cover_y_axis = travel_time(to_x,from_y,to_x,to_y,travel_speed_when_free)
            if time_since_last_free < time_needed_to_cover_x_axis: # if time enought to travel along x axis
                current_location_y = from_y
                current_location_x = interpolate(from_x, to_x,time_since_last_free, time_needed_to_cover_x_axis)
            else: 
                time_left_to_travel_along_y_axis = time_since_last_free - time_needed_to_cover_x_axis
                current_location_x = to_x
                current_location_y = interpolate(from_y, to_y,time_left_to_travel_along_y_axis, time_needed_to_cover_y_axis)

        return(current_location_x, current_location_y)

    def get_free_ambulance_id_n_loc(Current_time):
        global call_table
        global simulation_table
        global ambulance
        
        free_ambulance_id_n_loc_list = []
        for i in range(0,num_ambulance): 
            time_next_free = ambulance[i,0]
            if (time_next_free < Current_time): 
                x, y = current_location_of_free_ambulance(Current_time, i)
                free_ambulance_id_n_loc_list.append([i,x, y])

        return(free_ambulance_id_n_loc_list)

    def update_ambulance_table_for_new_call_assigned(ambulance_id,call_id, Current_time):
        global call_table
        global simulation_table
        global ambulance
        
        call_loc_x = call_table[call_table['call_id'] == call_id]['call_loc_x'].values[0]
        call_loc_y = call_table[call_table['call_id'] == call_id]['call_loc_y'].values[0]
        current_location_x, current_location_y = current_location_of_free_ambulance(Current_time, ambulance_id)
        travel_time_to_call_loc = travel_time(current_location_x, current_location_y,call_loc_x, call_loc_y, travel_speed_when_on_call)
        arrival_time = Current_time + travel_time_to_call_loc
        next_free_time = arrival_time + time_to_serve_at_call_loc()
        
        ambulance[ambulance_id,0] = next_free_time
        ambulance[ambulance_id,1] = call_table[call_table['call_id'] == call_id]['call_loc_x'].values[0]
        ambulance[ambulance_id,2] = call_table[call_table['call_id'] == call_id]['call_loc_y'].values[0]
        return(arrival_time, next_free_time)

    def f_x_y (x,y):
        z = 0.92*(1.6-abs(x-0.8)-abs(y-0.8))
        return(z)

    def simulate_call_location():
        M = 1.5
        x = round(random_state.uniform(0,1),5)
        y = round(random_state.uniform(0,1),5)
        z = round(random_state.uniform(0,1),5) 
        #print(x,y,z)

        while f_x_y(x,y) <= M*z:
            x = round(random_state.uniform(0,1),5)
            y = round(random_state.uniform(0,1),5) 
            z = round(random_state.uniform(0,1),5)

        return(x,y)

    def create_new_call_record(time_of_call):
        global call_table
        global simulation_table
        global ambulance
        time_created = time_of_call
        if np.shape(call_table)[0]== 0: ## First call of the simulation 
            new_call_id = 1       
        else:
            new_call_id = call_table.loc[call_table.call_id.idxmax(),:]['call_id']+1

        call_loc_x, call_loc_y = simulate_call_location()
        call_table = call_table.append({'call_id':new_call_id,'time_created':time_created, 'call_loc_x':call_loc_x, 'call_loc_y':call_loc_y}, ignore_index=True)
        #print(call_table)
        return(new_call_id)

    def update_call_attend_details(call_id, Current_time, time_ambulance_reached,time_completed,ambulance_id):
        global call_table
        global simulation_table
        global ambulance
        time_ambulance_assigned = Current_time
        ambulance_id_assigned = ambulance_id
        call_table.loc[call_table.call_id == call_id, ['time_ambulance_assigned', 'time_ambulance_reached', 'time_completed', 'ambulance_id_assigned']] = time_ambulance_assigned,time_ambulance_reached,time_completed, ambulance_id_assigned 

    def find_nearest_free_ambulance(Current_time, call_loc_x, call_loc_y):
        global call_table
        global simulation_table
        global ambulance
        free_ambulance_id_n_loc_list = get_free_ambulance_id_n_loc(Current_time)
        num_free_ambulance = np.shape(free_ambulance_id_n_loc_list)[0]
        ambulance_id = -1 #returns -1 is no ambulance is currently free. 

        if (num_free_ambulance > 0):  ## atleast one free ambulance currently
            selected_time = 100000
            for i in range(0,num_free_ambulance):  
                x1 = free_ambulance_id_n_loc_list[i][1]
                y1 = free_ambulance_id_n_loc_list[i][2]
                free_ambulance_id = free_ambulance_id_n_loc_list[i][0]
                time = travel_time(x1,y1,call_loc_x, call_loc_y,travel_speed_when_on_call) 
                if time < selected_time:
                    selected_time = time
                    ambulance_id = free_ambulance_id
        return(ambulance_id)

    def assign_ambulance_to_call(Current_time, call_id):
        global call_table
        global simulation_table
        global ambulance
        call_loc_x = call_table[call_table['call_id'] == call_id]['call_loc_x'].values[0]
        call_loc_y = call_table[call_table['call_id'] == call_id]['call_loc_y'].values[0]
        ambulance_id = find_nearest_free_ambulance(Current_time, call_loc_x, call_loc_y)
        time_ambulance_reached,time_completed = update_ambulance_table_for_new_call_assigned(ambulance_id,call_id, Current_time)
        update_call_attend_details(call_id, Current_time, time_ambulance_reached,time_completed,ambulance_id )
        return ()

    def time_of_next_completion(Current_time):
        global call_table
        global simulation_table
        global ambulance
    
        if (np.shape(simulation_table)[0]==0):
            time_when_next_call_completes = call_table.loc[0,:]['time_completed']

        elif np.shape(call_table[call_table['time_completed']>=Current_time])[0] == 0:
            time_when_next_call_completes = 100000
        else:
            time_when_next_call_completes = call_table[call_table['time_completed']>Current_time].time_completed.min()
        return(time_when_next_call_completes)

    def time_of_next_call(Current_time,lambda_call):
        global call_table
        global simulation_table

        time = Current_time + random_state.exponential(1/lambda_call)
        return(time)

    def time_of_next_call_assignment(Current_time):
        global call_table
        global simulation_table
        global ambulance
        free_ambulance_id_n_loc_list = get_free_ambulance_id_n_loc(Current_time)
        num_free_ambulance = np.shape(free_ambulance_id_n_loc_list)[0]
        time = 0

        if num_free_ambulance > 0:
            time = Current_time
        else:
            time = Current_time + time_of_next_completion

        return(time)

    def execute_current_event(Current_event,Current_time):
        global call_table
        global simulation_table
        global ambulance
        
        free_ambulance_id_n_loc_list = get_free_ambulance_id_n_loc(Current_time)
        num_free_ambulance = np.shape(free_ambulance_id_n_loc_list)[0]
        #print("Time to obtain free ambulance location and ids:",t2_get_free_ambu_detail-t1_get_free_ambu_detail)
        #print("number of free ambulances:", num_free_ambulance, "at time:",Current_time)

        if Current_event == "new_call_arrived":
            new_call_id = create_new_call_record(Current_time)
            #print(new_call_id)
            if(num_free_ambulance > 0):
                assign_ambulance_to_call(Current_time, new_call_id)
                #print("Time to execute new call arrived event:",t2_execute_new_call-t1_execute_new_call)
        else: #when Current event is "next_service_completes"
            unassigned_call_ids = call_table[pd.isnull(call_table['time_ambulance_assigned'])]
            if (np.shape(unassigned_call_ids)[0] >= 1):
                next_call_id_to_be_assigned = unassigned_call_ids.loc[unassigned_call_ids['time_created'].astype(float).idxmin(),:]['call_id']
                assign_ambulance_to_call(Current_time, next_call_id_to_be_assigned)
                 
    #Simulation run 
    Next_event_time = time_of_next_call(0, lambda_call)
    Next_event = "new_call_arrived" #initiating event list
    #event_list = {"new_call_arrived", "next_service_completes"}

    while Next_event_time < T_max:
        Current_time = Next_event_time
        Current_event = Next_event
        
        execute_current_event(Current_event,Current_time)
        
        if Current_event == "new_call_arrived":
            Time_of_next_call = time_of_next_call(Current_time,lambda_call)
        else:
            Time_of_next_call = simulation_table.iloc[-1,:]['Time_of_next_call']
        
        Time_of_next_call_completed = time_of_next_completion(Current_time)
        
        if ((Time_of_next_call < Time_of_next_call_completed) or (np.isnan(Time_of_next_call_completed))): #either next call will occur before next completion or all calls are complete. 
            Next_event = "new_call_arrived"
            Next_event_time = Time_of_next_call
        else: 
            Next_event = "next_service_completes"
            Next_event_time = Time_of_next_call_completed
        
        simulation_table = simulation_table.append({'Current_time':Current_time, 'Current_event':Current_event, 'Time_of_next_call':Time_of_next_call, 'Time_of_next_call_completed':Time_of_next_call_completed, 'Next_event':Next_event, 'Next_event_time':Next_event_time} , ignore_index=True)
    calls_served = call_table[(~pd.isnull(call_table['time_ambulance_reached'])) & (call_table['time_created'] >= cool_off_period)] 
    response_times = np.atleast_1d(calls_served['time_ambulance_reached']-calls_served['time_created']) 
    return(response_times)

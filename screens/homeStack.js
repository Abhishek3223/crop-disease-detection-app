
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
import ShowData from './showData';
import HomePage from './homePage';
import UserInfo from './userInfo';
import Signup from './signup';
import Comments from './comments';
import DroneData from './droneData';


const Stack = createStackNavigator();


export const DataStack = () => {
    return(
    <Stack.Navigator>
            <Stack.Screen name="SignIn" component={Signup} options={{ headerShown: false }}/>
            <Stack.Screen name="User_Info" component={UserInfo} options={{ headerShown: false }}/>
            <Stack.Screen name="HomePage" component={HomePage} options={{ headerShown: false }}/>
            <Stack.Screen name="ShowData" component={ShowData} options={{ headerShown: false }}/>
            <Stack.Screen name="Comments" component={Comments} options={{ headerShown: false }}/>
            <Stack.Screen name="DroneData" component={DroneData} options={{ headerShown: false }}/>
    </Stack.Navigator>
    );
}
export default function HomeStack() {
    return (
        <NavigationContainer>
            <DataStack />
        </NavigationContainer>
    );
}
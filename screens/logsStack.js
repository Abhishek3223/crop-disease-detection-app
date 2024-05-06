import {View} from "react-native"
import { createStackNavigator } from '@react-navigation/stack';
import Comments from './comments';
import Logs from './logs';
import { NavigationContainer } from "@react-navigation/native";
import ShowData from "./showData";


const Stack = createStackNavigator();

export const CommentStack=()=>{
    return (
        <Stack.Navigator>
            <Stack.Screen name="Logs" component={Logs} options={{ headerShown: false }}/>
            <Stack.Screen name="Comments" component={Comments} options={{ headerShown: false }}/>
        </Stack.Navigator>
    );
}
export default function LogsStack(){
    return (
        <NavigationContainer>
            <CommentStack/>
        </NavigationContainer>
    )
}
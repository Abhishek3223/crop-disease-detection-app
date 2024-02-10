import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
import Upload from './uploadPage';
import GetDetails from './getDetails';

const Stack = createStackNavigator();

export const UploadStack = () => {
    return(
    <Stack.Navigator>
            <Stack.Screen name="Camera" component={Upload} options={{ headerShown: false }}/>
            <Stack.Screen name="Details" component={GetDetails} options={{ headerShown: false }}/>
    </Stack.Navigator>
    );
}
export default function CamStack() {
    return (
        <NavigationContainer>
            <UploadStack/>
        </NavigationContainer>
    );
}
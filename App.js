import { StyleSheet, Text, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import ProfilePage from './screens/profilePage';
import { DataStack } from './screens/homeStack';
import { UploadStack } from './screens/camStack';
import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { AuthProvider, useAuth } from './screens/context';
import {CommentStack} from './screens/logsStack';

const Tab = createBottomTabNavigator();

const MainTabs = () => {
  const { user } = useAuth();

  return (
    <NavigationContainer style={styles.container}>
      <Tab.Navigator>
        <Tab.Screen name="Home" component={DataStack} options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="home-outline" color="#50c878" size={size} /> // Change the name to the desired icon
          ),
        }} />

        {user && ( // Conditionally render the "Logs" tab
          <>
          {user.userType === "Expert" && (
            <Tab.Screen name="Posts" component={CommentStack} options={{
              tabBarIcon: ({ color, size }) => (
                <Ionicons name="people-outline" color="#50c878" size={size} /> // Change the name to the desired icon
              ),
            }} />
          )}
          {user.userType === "Farmer" && (
            <Tab.Screen name="Upload" component={UploadStack} options={{
              tabBarIcon: ({ color, size }) => (
                <Ionicons name="camera-outline" color="#50c878" size={size} /> // Change the name to the desired icon
              ),
            }} />
          )}
            <Tab.Screen name="Profile" component={ProfilePage} options={{
              tabBarIcon: ({ color, size }) => (
                <Ionicons name="settings-outline" color="#50c878" size={size} /> // Change the name to the desired icon
              ),
            }} />
          </>
        )}
      </Tab.Navigator>
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainTabs />
    </AuthProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});

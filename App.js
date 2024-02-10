
import { StyleSheet, Text, View } from 'react-native';
import {NavigationContainer} from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import ProfilePage from './screens/profilePage';
import {DataStack} from './screens/homeStack';
import {UploadStack} from './screens/camStack';
import { firebase } from '@react-native-firebase/auth';

const Tab = createBottomTabNavigator();





export default function App() {
  return (
    <NavigationContainer style={styles.container}>
      <Tab.Navigator>
        <Tab.Screen name="Home" component={DataStack} />
        <Tab.Screen name="Upload" component={UploadStack} />
        <Tab.Screen name="Profile" component={ProfilePage} />
      </Tab.Navigator>
    </NavigationContainer>
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

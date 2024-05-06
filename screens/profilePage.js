import { StyleSheet, Text, View, Button, Image } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { useAuth } from "./context";
import auth from '@react-native-firebase/auth';
import { useNavigation } from '@react-navigation/native';

export default function ProfilePage() {
  const { user, setUser } = useAuth();
  const navigation = useNavigation();

  const logout = async () => {
    try {
      await auth().signOut();
      setUser(null);
      //reseting the vaigation stack to 0
      navigation.reset({
        index: 0,
        routes: [{ name: 'SignIn' }],
      })
    }
    catch (e) {
      console.log("Error logging out: ", e)
    }
  }
  return (
    <View style={styles.container}>
      <View style={styles.prof_content}>
        <Image
          style={styles.pic}
          source={require('../assets/prof_pic.webp')}
          size={20} 
        />
        <Text style={styles.name}>{user.username}</Text>
      </View>
      <View style={styles.button}>
        <Button color="#50c878" onPress={logout} title='Log Out'/>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  button: {
    width: 100,
    paddingVertical: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  pic: {
    width: 200,
    height: 200,
    borderRadius: 100,
    marginBottom: 20,
  },
  prof_content: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
  },
  name: {
    fontSize: 20,
    fontWeight: 'medium',
  },

});
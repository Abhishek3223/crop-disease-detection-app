import { StyleSheet, Text, View, Image, Button } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { useNavigation } from '@react-navigation/native';
import Ionicons from '@expo/vector-icons/Ionicons'
import auth from '@react-native-firebase/auth';
import { useAuth } from './context';

export default function HomePage() {

    const { user } = useAuth();
    const navigation = useNavigation();

    const handleClick=()=>{
        if(user.userType=='Expert'){
            navigation.navigate('Logs')
        }
        else{
            navigation.navigate('ShowData')
        }
    }
    
    return (
        <View style={styles.container}>
            <View>
                <Image
                    style={styles.plantImage}
                    source={require('../assets/plant_pic.jpg')}
                />
            </View>
            <View >
                <Text style={styles.PlantText}>We Care for your Needs</Text>
            </View>
            <View style={styles.btnSection}>
                <Button
                    style={styles.btn}
                    onPress={handleClick}
                    title="Get Farm"
                    color="#50c878"
                    accessibilityLabel="Learn more about this purple button"
                />
                <View style={styles.droneButton}>
                <Button
                    style={styles.btn}
                    onPress={() => {navigation.navigate('DroneData')}}
                    title="Connect to Drone"
                    color="#50c878"
                    accessibilityLabel="Learn more about this purple button"
                />
                </View>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'flex-start'
    },
    plantImage: {
        marginVertical: 40,
        width: 200,
        height: 300,
    },
    plantText: {
        fontSize: 40,
        fontWeight: 'bold',
    },
    btnSection: {
        width: 200,
        paddingVertical: 20,
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    btn: {
        width: 40,
        borderRadius: 20,
        color: '#50c878'
    },
    // icon: {
    //     justifyContent:'space-between'
    // },
    iconSection: {
        justifyContent: 'flex-end',
        alignSelf: 'flex-end',
        padding: 10
    },
    droneButton: {
        marginTop: 20
    }
});
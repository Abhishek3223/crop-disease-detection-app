import { Text, View, Image, Button, StyleSheet, TextInput } from 'react-native';
import React from 'react';
import { Picker } from '@react-native-picker/picker';
import firestore from '@react-native-firebase/firestore';

export default function UserInfo({ route, navigation }) {

    const [username, setUsername] = React.useState('');
    const [cropType, setCropType] = React.useState('');
    const { uid } = route.params;
    const crops = ["Maize", "Rice", "Wheat", "Bajra"]

    function saveDetails() {
        try{
            firestore().collection('users').doc(uid).set({
                username: username,
                cropType: cropType
            })  
            navigation.navigate('HomePage');
        }
        catch(e){
            console.log("Error saving details: ", e);
        }
    };
    

    return (
        <View style={styles.container}>
            <Text style={styles.header}>Complete your Profile</Text>
            <View>
                <TextInput
                    style={styles.input}
                    onChangeText={onChangeText = (text) => { setUsername(text) }}
                    placeholder="Username"
                    keyboardType="default"
                    importantForAccessibility='yes'
                />
                <Picker
                style={styles.select}
                    selectedValue={cropType}
                    onValueChange={(itemValue, itemIndex) =>
                        setCropType(itemValue)
                    }
                    importantForAccessibility='yes'
                    >
                    <Picker.Item label="Crop Type" style={styles.Item} value="" enabled={false} />
                    {crops.map((crop, index)=>{
                        return <Picker.Item label={crop} value={crop} key={index}/>
                    })}
                    
                </Picker>
                <Button
                    title="Ready to go"
                    color="#50c878"
                    onPress={() => { saveDetails()}}
                />
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
    input: {
        height: 50,
        margin: 12,
        borderWidth: 1,
        borderRadius: 100,
        borderColor: '#50c878',
        padding: 10,
        width: 200
    },
    header: {
        paddingVertical: 25,
        fontSize: 18,
        fontWeight: 'medium',
        alignSelf: 'center'
    },
    Item:{
        color: 'grey',
        fontSize: 15,
    },
    select:{
        height: 50,
        margin: 12,
        borderWidth: 1,
        borderRadius: 100,
        padding: 10,
        width: 200,
        color: 'black'
    }
});
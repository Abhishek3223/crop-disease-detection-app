import React from 'react';
import { View, StyleSheet, Image, Text, TextInput, Button } from 'react-native';
import firestore from '@react-native-firebase/firestore';
import { useAuth } from './context';
import { useNavigation } from '@react-navigation/native';


export default function GetDetails({ route }) {
    const { user, setUser } = useAuth();
    const photo = route.params.photo;
    const [name, setName] = React.useState('');
    const navigation = useNavigation();

    const handleClick = async () => {
        try {
            console.log("user",user)
            // Create a new document in the "photos" collection
            console.log("routeparams", route.params);
            await firestore().collection('photos').doc(photo.id).set({
                userId: user.uid, // Assuming userId is available in the user object
                postId: photo.id, // You need to get the postId from somewhere in your app
                photoUri: photo.uri,
                cropName: name // Add the crop name
            });
            navigation.navigate('ShowData');
        } catch (e) {
            console.log("Error adding photo: ", e);
        }
    }

    return (
        <View style={styles.container}>
            <View>
                <Text style={styles.header}>Your image is ready to be Inspected👩‍🔬!</Text>
                <View style={styles.card}>
                    <Image source={{ uri: photo && photo.uri }} style={styles.image} />
                    <TextInput
                        style={styles.input}
                        value={name}
                        onChangeText={text => setName(text)}
                        placeholder="Enter the name of the crop"
                        keyboardType="default"
                    />
                    <Button title="submit" color="#50c878" style={styles.button} onPress={handleClick} />
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
        justifyContent: 'flex-start',
    },
    card: {
        padding: 20,
        borderRadius: 20,
        backgroundColor: '#E4F5E3',
        alignItems: 'center',
        justifyContent: 'center',
    },
    image: {
        width: 250,
        height: 200,
    },
    header: {
        paddingVertical: 25,
        fontSize: 18,
        fontWeight: 'medium',
        alignSelf: 'center'
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
    button: {
        width: 30,
        borderRadius: 20,
    }
});

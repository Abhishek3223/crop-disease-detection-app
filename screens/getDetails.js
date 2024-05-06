import React from 'react';
import { View, StyleSheet, Image, Text, TextInput, Button, Alert } from 'react-native';
import { useAuth } from './context';
import { useNavigation } from '@react-navigation/native';
import firestore from '@react-native-firebase/firestore';
import { firebase } from '@react-native-firebase/auth';

export default function GetDetails({ route }) {
    const { user } = useAuth();
    const photo = route.params.photo;
    const [name, setName] = React.useState('');
    const navigation = useNavigation();

    const handleClick = async () => {
        try {
            console.log(photo)
            const formData = new FormData();
            formData.append('image', {
                uri: photo.uri,
                type: 'image/jpeg', // or whatever the MIME type of your image is
                name: 'photo.jpg' // or any name you want to give to the file
            });

            const response = await fetch('http://192.168.156.55:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                body: formData,
            });
            const data = await response.json();
            console.log(data);

            // Handle response from Flask server
            // For example, you can display a success message
            const cropNames = ['common rust', 'grey leaf spot', 'healthy', 'northern leaf blight'];
            const prediction = [];
            data.prediction.forEach((value, index) => {
                if (value === 1 && cropNames[index]) {
                    prediction.push(cropNames[index]);
                }
            });

            console.log(prediction);
            firebase.firestore().collection('photos').doc(photo.id).set({
                userId: user.uid,
                photoUri: photo.uri,
                postId: photo.id,
                cropName: name,
                prediction: prediction,
            });
            Alert.alert('Success', 'Image submitted successfully', [{ text: 'OK', onPress: () => navigation.navigate('ShowData') }]);
        } catch (error) {
            console.error('Error submitting data:', error);
            // Handle error
            Alert.alert('Error', 'An error occurred while submitting image', [{ text: 'OK' }]);
        }
    };

    return (
        <View style={styles.container}>
            <View>
                <Text style={styles.header}>Your image is ready to be Inspectedr👩‍🔬!</Text>
                <View style={styles.card}>
                    <Image source={{ uri: photo && photo.uri }} style={styles.image} />
                    <TextInput
                        style={styles.input}
                        value={name}
                        onChangeText={text => setName(text)}
                        placeholder="Enter the name of the crop"
                        keyboardType="default"
                    />
                    <Button title="Submit" color="#50c878" onPress={handleClick} />
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
});

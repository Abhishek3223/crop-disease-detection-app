import { View, Text, StyleSheet, ScrollView, Image } from 'react-native';
import React from 'react';
import { Card, Button, Icon } from '@rneui/themed'; // Import Card and Button components
import { useAuth } from './context';
import firestore from '@react-native-firebase/firestore';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useNavigation } from '@react-navigation/native';
import { Dimensions } from 'react-native';

const windowWidth = Dimensions.get('window').width;

export default function ShowData() {
    const { user } = useAuth();
    const [loading, isLoading] = React.useState(true);
    const [images, setImages] = React.useState([]);
    const navigation = useNavigation();

    React.useEffect(() => {
        const fetchUserPhotos = async () => {
            try {
                const photosSnapshot = await firestore()
                    .collection('photos')
                    .where('userId', '==', user.uid)
                    .get();
                const userPhotos = []; // Initialize an array to store photos
                photosSnapshot.forEach(doc => {
                    // Push the document data to the userPhotos array
                    userPhotos.push(doc.data());
                });
                setImages(userPhotos); // Set the images state with the photos array
                isLoading(false);
            } catch (e) {
                console.log("Error getting user photos: ", e);
            }
        };

        fetchUserPhotos();
    }, []);

    return (
        <ScrollView>
            <View style={styles.container}>
                {images.map((photoData, index) => (
                    <Card key={index}>
                        <Card.Title>{photoData.cropName}</Card.Title>
                        <Card.Divider />
                        <Card.Image
                            style={{ padding: 0 }}
                            source={{ uri: photoData.photoUri }} // Set the source of the image
                        />
                        <Text style={styles.text}>
                            Result: {photoData.prediction ? photoData.prediction.join(', ') : 'No prediction available'}
                        </Text>

                        <View style={styles.commentSection}>

                            <Ionicons
                                name="chatbox-ellipses-outline"
                                color="#50c878"
                                size={32}
                                iconStyle={{ alignItems: 'center' }}
                                onPress={() => navigation.navigate('Comments', { postId: photoData.id, comments: photoData.comments })}
                            />
                        </View>
                    </Card>
                ))}
            </View>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        margin: 20
    },
    text: {
        fontSize: 16,
        paddingVertical: 10,
        fontWeight: 400,
        width: windowWidth * 0.8
    },
    card: {
        display: 'flex',
        alignItems: 'center',
    }
});

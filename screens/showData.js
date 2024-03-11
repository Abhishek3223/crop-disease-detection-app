import { View, Text, StyleSheet, ScrollView, Image } from 'react-native';
import React from 'react';
import { Card, Button, Icon } from '@rneui/themed'; // Import Card and Button components
import { useAuth } from './context';
import firestore from '@react-native-firebase/firestore';

export default function ShowData() {
    const { user } = useAuth();
    const [loading, isLoading] = React.useState(true);
    const [images, setImages] = React.useState([]);

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
                        <Card.Divider/>
                        <Card.Image
                            style={{padding: 0}}
                            source={{ uri: photoData.photoUri }} // Set the source of the image
                        />
                        <Text style={{marginBottom: 10}}>
                            The idea with React Native Elements is more about component structure than actual design.
                        </Text>
                        <Button
                            icon={
                                <Icon
                                    name="code"
                                    color="#ffffff"
                                    iconStyle={{marginRight: 10}}
                                />
                            }
                            buttonStyle={{
                                borderRadius: 0,
                                marginLeft: 0,
                                marginRight: 0,
                                marginBottom: 0
                            }}
                            title="VIEW NOW"
                        />
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
        margin: 20
    }
});

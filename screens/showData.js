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
        const fetchUser = async () => {
            try {
                const usersSnapshot = await firestore().collection('users').doc(user.username).get();
                const userData = usersSnapshot.data();
                const userPhotos = userData.photos || []; // Ensure photos array exists
                setImages(userPhotos); // Set the images state with the photos array
                isLoading(false);
            } catch (e) {
                console.log("Error getting user data: ", e);
            }
        };
        fetchUser();
    }, []);

    return (
        <ScrollView>
            <View style={styles.container}>
                {images.map((photoUri, index) => ( // Map over the images array
                    <Card key={index}>
                        <Card.Title>Image {index + 1}</Card.Title>
                        <Card.Divider/>
                        <Card.Image
                            style={{padding: 0}}
                            source={{ uri: photoUri }} // Set the source of the image
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

import React from 'react';
import { StyleSheet, View, Image, ScrollView, Dimensions } from 'react-native';
import { Text, Card, Button, Icon } from '@rneui/themed';
import firestore from '@react-native-firebase/firestore';
import { useAuth } from './context';
import { useNavigation } from '@react-navigation/native';
import Ionicons from '@expo/vector-icons/Ionicons';

const windowWidth = Dimensions.get('window').width;

export default function Logs() {

    const [loading, isLoading] = React.useState(true);
    const [users, setUsers] = React.useState([]);
    const [comments, setComments] = React.useState({});
    const navigation = useNavigation();

    React.useEffect(() => {
        const fetchUsers = async () => {
            try {
                const usersSnapshot = await firestore().collection('photos').get();
                const usersList = usersSnapshot.docs.map(doc => ({
                    id: doc.id,
                    ...doc.data()
                }));
                setUsers(usersList);
                console.log("Userslist: ", usersList);
                isLoading(false);
            }
            catch (e) {
                console.log("Error getting users: ", e);
            }
        };
        fetchUsers();
    }, []);
    const handleCommentChange = (postId, comment) => {
        setComments(prevComments => ({
            ...prevComments,
            [postId]: comment // Store the comment text with the postId as the key
        }));
    };

    return (
        <ScrollView>
            <View style={styles.container}>
                {users.map((post, index) => (
                    <Card key={index} style={styles.card}>
                        <Card.Title>{post.cropName}</Card.Title>
                        <Card.Divider />
                        <Card.Image
                            style={{ padding: 0}}
                            source={{
                                uri: post.photoUri // Replace this with the actual photo
                            }}
                        />
                        <Text style={styles.text}>
                            Result: {post.prediction ? post.prediction.join(', ') : 'No prediction available'}
                        </Text>
                        <View style={styles.commentSection}>
                            <Ionicons
                                name="chatbox-ellipses-outline"
                                color="#50c878"
                                size={32}
                                iconStyle={{ alignItems: 'center', justifyContent: 'center'}}
                                onPress={() => navigation.navigate('Comments', { postId: post.id, comments: post.comments })}
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
        margin: 0
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
    },
});

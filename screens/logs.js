import React from 'react';
import { StyleSheet, View, Image, ScrollView} from 'react-native';
import { Text, Card, Button, Icon } from '@rneui/themed';
import firestore from '@react-native-firebase/firestore';
export default function Logs() {

    const [loading, isLoading]=React.useState(true);
    const [users, setUsers]=React.useState([]);

    React.useEffect(()=>{
        const fetchUsers=async()=>{
        try{
            const usersSnapshot= await firestore().collection('users').get();
            const usersList= usersSnapshot.docs.map(doc=>({
                id: doc.id,
                ...doc.data()
            }));
            setUsers(usersList);
            console.log("Users: ", users);
            isLoading(false);
        }
        catch(e){
            console.log("Error getting users: ", e);
        }
    };
        fetchUsers();
    }, []);

    return (
        <ScrollView>
        <View style={styles.container}>
            {users.map((user, index)=>(
                    <Card key={index}>
                        <Card.Title>{user.username}</Card.Title>
                        <Card.Divider/>
                        <Card.Image
                            style={{padding: 0}}
                            source={{
                                uri: user.photo // Replace this with the actual photo
                            }}
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
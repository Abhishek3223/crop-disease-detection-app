import {View, Text, StyleSheet} from 'react-native';



export default function ShowData() {
    return (
        <View style={styles.container}>
            <Text>Hello, We Studied your Farm!</Text>
            <Text>Here are the results</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
    }
});
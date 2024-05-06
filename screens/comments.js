import { View, StyleSheet, Text, TextInput } from "react-native"
import { useAuth } from "./context"
import React, { useEffect } from "react"
import { firebase } from "@react-native-firebase/firestore";
import Ionicons from '@expo/vector-icons/Ionicons';

export default function Comments({ route }) {
    const inputRef = React.useRef();
    let [comments, setComments] = React.useState({});
    const [comment, setComment] = React.useState('');
    const { user, setUser } = useAuth();
    const postId = route.params.postId;
    comments = route.params.comments || [];
    const [loading, isLoading] = React.useState(true);
    const [postData, setPostData] = React.useState({});



    const postComment = async() => {
        let allComments = comments || [];
        console.log(comment,"commentttt")
        allComments.push({ comment: comment, userId: user.uid, postId: postId, userName: user.username});
        // console.log(allComments);
        try {
            console.log(postId,"posttttiiddiidd")
            firebase.firestore().collection('photos').doc(postId).update({
                comments: allComments
            })
            await getNewComments();
        }
        catch (e) {
            console.log("Error posting comment: ", e);
        }
        inputRef.current.clear();
    }
    const getNewComments=async()=>{
        let commentsdata=[]
        const data = await firebase.firestore().collection('photos').doc(postId).get()
        .then(doc=>{
        //    setComments(doc.data().comments)
        commentsdata=doc.data().comments
        })
        setComments(commentsdata);
        console.log(data,"data")
    }

    return (
        <View style={styles.container}>
            <Text>Comments</Text>
            {comments.length > 0 ? comments.map((comment, index) => (
                <View key={index} style={styles.displayComments}>
                    <Ionicons name="person-circle-outline" size={32} color="green" />
                    <View style={styles.commentContent}>
                    <Text style={styles.commentName}>{comment.userName}</Text>
                    <Text>{comment.comment}</Text>
                    </View>
                </View>
            )) : null}
            <View style={styles.commentSection}>
                <TextInput
                    ref={inputRef}
                    style={styles.input}
                    value={comment}
                    onChangeText={onChangeText = (text) => { setComment(text) }}
                    placeholder="Add a comment"
                    keyboardType="default"
                />
                <Ionicons name="send" size={25} color="green" onPress={postComment} />
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        flexDirection: 'flex-end',
        alignItems: 'center',
    },
    commentSection: {
        position: 'absolute',
        bottom: 0,
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'center',
        width: '100%',
        paddingBottom: 10
    },
    input: {
        borderColor: 'grey',
        flexWrap: 'wrap',
        width: '75%',
        borderRadius: 30,
    },
    displayComments: {
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        marginLeft: 10,
        padding: 10,
        width: '100%'
    },
    commentName: {
        fontSize: 12,
        color: 'grey',
    },
    commentContent: {
        display: 'flex',
        flexDirection: 'column',
        marginLeft: 10
    }
})
$(function() {
    $('#form').unbind('submit').submit(function() {
        event.preventDefault();
        var form_data = new FormData($('#form')[0]);
        console.log(form_data);
        $.ajax({
            type: 'POST',
            url: '/get_score',
            data: form_data,
            contentType: false,
            processData: false,
            dataType: 'json'
        }).done(function(data){
            console.log(data);
            console.log('Success!');
            alert(data.result + '\n' + 'The face distance is: ' + data.score)
        }).fail(function(data){
            alert('error!');
        });
        return false;
    });
}); 
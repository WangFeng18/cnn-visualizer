import React from 'react';

class Image extends React.Component{
    constructor(props){
        super(props)
    }

    render(){
        return(
            // <image src={require(this.props.add)}>
            <img onMouseOver={this.props.onmouseover} className={this.props.class} src={this.props.add} width={this.props.width} />
        )
    }
}
export default Image;
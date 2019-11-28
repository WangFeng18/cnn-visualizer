import React from 'react';
import Image from './Image';

class Gallery extends React.Component{
    constructor(props){
        super(props);
        this.paths = require('../paths/filters_vis.json');
        this.all_layers = Object.keys(this.paths);
        this.state = {
            gallery_rows: 8,
            show_layer: this.all_layers[0],
            activate_img: 0,
            path: require('../paths/filters_vis.json'),
        }
    }

    changelayer(i){
        this.setState({
            show_layer: this.all_layers[i]
        })
    }

    diszoom(i){
        this.setState({
            activate_img:i
        })
    }

    show_feature_map(){
        this.setState({
            path: require('../paths/feature_map.json')
        })
    }

    show_filters(){
        this.setState({
            path: require('../paths/filters_vis.json')
        })
    }

    render(){
        let gallery_data = new Array()
        for (var i = 0; i < this.state.path[this.state.show_layer].length; i++) { 
            gallery_data[i] = <Image onmouseover={this.diszoom.bind(this, i)} class='gallery_element' add={require('../'+this.state.path[this.state.show_layer][i])} width={String(100/this.state.gallery_rows-0.2)+'%'} key={String(i)} />
        }

        let buttons = []
        for (var i=0; i<this.all_layers.length; i++) {
            buttons[i] = <button key={i} onClick={this.changelayer.bind(this, i)}>{this.all_layers[i]}</button>
        }

        return(
            <div id='root'>
                <div id='navigator'>
                    <button onClick={this.show_feature_map.bind(this)}>Feature Map</button>
                    <button onClick={this.show_filters.bind(this)}>Filter Visulization</button>
                </div>
                <div id='bigImage' align='center'>
                    <Image add={require('../imgs/source_image/A.JPEG')} width='100%'/>
                    <Image add={require('../'+this.state.path[this.state.show_layer][this.state.activate_img])} width='100%'/>
                </div>

                <div id='gallery'>
                    <div id='button_panel'>
                        {buttons}       
                    </div>
                    <div id='gallery_body'>
                        {gallery_data}
                    </div>
                </div>
            </div>
        )
    }
}
export default Gallery;
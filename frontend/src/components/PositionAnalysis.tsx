import React from 'react';
import { Card, List, Tag, Typography } from 'antd';
import { CheckCircleOutlined, ExclamationCircleOutlined, QuestionCircleOutlined } from '@ant-design/icons';

const { Text } = Typography;

interface AnalysisProps {
  analysis?: {
    status: string;
    group_size: number;
    eyes: number;
    liberties: number;
  };
  joseki?: string[];
  vitalPoints?: [number, number][];
}

const PositionAnalysis: React.FC<AnalysisProps> = ({ analysis, joseki, vitalPoints }) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'alive':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'in_danger':
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      default:
        return <QuestionCircleOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'alive':
        return '活棋';
      case 'in_danger':
        return '危险';
      case 'likely_alive':
        return '可能活';
      default:
        return '未知';
    }
  };

  return (
    <Card title="位置分析" size="small" style={{ marginTop: 16 }}>
      {analysis && (
        <>
          <div style={{ marginBottom: 16 }}>
            <Text strong>状态：</Text>
            {getStatusIcon(analysis.status)}{' '}
            <Text>{getStatusText(analysis.status)}</Text>
          </div>
          <div style={{ marginBottom: 16 }}>
            <Text>棋块大小：{analysis.group_size} 子</Text>
            <br />
            <Text>眼位数量：{analysis.eyes}</Text>
            <br />
            <Text>气数：{analysis.liberties}</Text>
          </div>
        </>
      )}
      
      {joseki && joseki.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Text strong>匹配定式：</Text>
          <br />
          {joseki.map((pattern, index) => (
            <Tag key={index} color="blue" style={{ margin: '4px' }}>
              {pattern}
            </Tag>
          ))}
        </div>
      )}

      {vitalPoints && vitalPoints.length > 0 && (
        <div>
          <Text strong>关键点：</Text>
          <List
            size="small"
            dataSource={vitalPoints}
            renderItem={(point) => (
              <List.Item>
                {`(${point[0] + 1}, ${point[1] + 1})`}
              </List.Item>
            )}
          />
        </div>
      )}
    </Card>
  );
};

export default PositionAnalysis;
